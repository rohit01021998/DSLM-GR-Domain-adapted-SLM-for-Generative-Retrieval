import re
import json
import torch
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

from genret import config
from genret.trie import IDTrie, load_trie, make_prefix_allowed_tokens_fn
from genret.embed import load_chunks_jsonl
from genret.baselines import BM25Retriever

class PAGLogitsProcessor(LogitsProcessor):
    """
    SIGIR 2024 Planning-Ahead Generative Retrieval (PAG) Logits Processor.
    Injects subtree lookahead maximum prior scores into prefix candidate logits:
        s'(c <= i) = s(c <= i) + alpha * max_{d in Subtree(c <= i)} s^simul(q, d)
    """
    def __init__(
        self,
        trie: IDTrie,
        digit_token_ids: Dict[int, int],
        id_start_token_id: int,
        id_end_token_id: int,
        doc_priors: Dict[str, float],
        alpha: float = 2.0
    ):
        self.trie = trie
        self.digit_token_ids = digit_token_ids
        self.token_to_digit = {tok_id: digit for digit, tok_id in digit_token_ids.items()}
        self.id_start_token_id = id_start_token_id
        self.id_end_token_id = id_end_token_id
        self.doc_priors = doc_priors
        self.alpha = alpha

        # Precompute subtree max priors across all trie prefixes once per query (Task 10d)
        self.prefix_max_priors: Dict[tuple, float] = {}
        def populate(node, prefix_tuple):
            if node.doc_ids:
                self.prefix_max_priors[prefix_tuple] = max(self.doc_priors.get(cid, 0.0) for cid in node.doc_ids)
            for digit, child in node.children.items():
                populate(child, prefix_tuple + (digit,))
        populate(self.trie.root, ())

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size_beam, vocab_size = scores.shape

        for b in range(batch_size_beam):
            ids_list = input_ids[b].tolist()

            # If opening <id> not yet generated, no digit prefix exists
            if self.id_start_token_id not in ids_list:
                continue

            start_idx = len(ids_list) - 1 - ids_list[::-1].index(self.id_start_token_id)
            generated_id_tokens = ids_list[start_idx + 1:]

            # If </id> already generated, skip
            if self.id_end_token_id in generated_id_tokens:
                continue

            # Extract generated digits
            current_digits = []
            for tok in generated_id_tokens:
                if tok in self.token_to_digit:
                    current_digits.append(self.token_to_digit[tok])
                else:
                    break

            # O(1) lookahead prior scoring
            curr_tuple = tuple(current_digits)
            allowed_digits = self.trie.allowed_next(current_digits)
            for d in allowed_digits:
                if d == -1:
                    max_prior = self.prefix_max_priors.get(curr_tuple, 0.0)
                    scores[b, self.id_end_token_id] += self.alpha * max_prior
                elif d in self.digit_token_ids:
                    tok_id = self.digit_token_ids[d]
                    next_tuple = curr_tuple + (d,)
                    max_prior = self.prefix_max_priors.get(next_tuple, 0.0)
                    scores[b, tok_id] += self.alpha * max_prior

        return scores

# Authoritative Euro NCAP Domain Glossary to prevent false acronym expansions
EURO_NCAP_GLOSSARY = {
    "CPFA": "Car-to-Pedestrian Farside Adult (CPFA)",
    "CPTA": "Car-to-Pedestrian Turning Adult (CPTA)",
    "CPLA": "Car-to-Pedestrian Longitudinal Adult (CPLA)",
    "CBTA": "Car-to-Bicyclist Turning Across (CBTA)",
    "CBLA": "Car-to-Bicyclist Longitudinal Adult (CBLA)",
    "CCRb": "Car-to-Car Rear Braking (CCRb)",
    "CCRs": "Car-to-Car Rear Stationary (CCRs)",
    "CCFos": "Car-to-Car Frontal Opposite Scenario (CCFos)",
    "CCFtap": "Car-to-Car Frontal Turn Across Path (CCFtap)",
    "CCCscp": "Car-to-Car Crossing Straight Crossing Path (CCCscp)",
    "CMRs": "Car-to-Motorcyclist Rear Stationary (CMRs)",
    "CMRb": "Car-to-Motorcyclist Rear Braking (CMRb)",
    "VUT": "Vehicle Under Test (VUT)",
    "GVT": "Global Vehicle Target (GVT)",
    "EPTa": "Euro NCAP Pedestrian Target adult (EPTa)",
    "EBTa": "Euro NCAP Bicyclist Target adult (EBTa)",
    "EMT": "Euro NCAP Motorcyclist Target (EMT)",
    "DTLE": "Distance To Lane Edge (DTLE)",
    "PBC": "Peak Braking Coefficient (PBC)",
    "SLIF": "Speed Limit Information Function (SLIF)",
    "iACC": "Intelligent Adaptive Cruise Control (iACC)",
    "SOV": "Secondary Other Vehicle (SOV)",
    "BSM": "Blind Spot Monitoring (BSM)",
    "LKA": "Lane Keeping Assist (LKA)",
    "ELK": "Emergency Lane Keeping (ELK)",
    "FCW": "Forward Collision Warning (FCW)",
    "AEB": "Autonomous Emergency Braking (AEB)",
    "ESS": "Emergency Steering Support (ESS)",
    "TTC": "Time To Collision (TTC)",
    "CSSI": "Continuous System Status Indicator (CSSI)",
    "DSM": "Driver State Monitoring (DSM)",
    "Vehicle width": "Vehicle width widest point of the vehicle ignoring rear-view mirrors",
    "Driver Intention Monitoring": "Driver Intention Monitoring distinguishing intentional from unintentional lane crossing"
}

def rewrite_query_for_retrieval(
    query: str,
    base_url: str = config.GEN_BASE_URL,
    model_name: str = None
) -> str:
    """
    Rewrites user query to optimize hybrid neural and lexical retrieval.
    Expands verified Euro NCAP abbreviations using the domain glossary,
    strictly preserves technical acronyms and exact numbers, and prevents hallucinated expansions.
    """
    if model_name is None:
        model_name = config.get_active_vllm_model(base_url)

    # If query is short greeting or command, keep as-is
    trimmed = query.strip()
    if len(trimmed.split()) <= 2 and trimmed.lower() in ["hi", "hello", "hey", "test", "help", "exit", "quit"]:
        return query

    # Check for known glossary terms in query to provide exact grounding
    glossary_hints = []
    for acronym, full_def in EURO_NCAP_GLOSSARY.items():
        # Match whole word
        if re.search(rf"\b{re.escape(acronym)}\b", trimmed, re.IGNORECASE):
            glossary_hints.append(f"{acronym} -> {full_def}")

    glossary_prompt_text = ""
    if glossary_hints:
        glossary_prompt_text = "Verified Domain Glossary:\n" + "\n".join(glossary_hints) + "\n"

    # Build fallback grounded search query with verified glossary expansions
    glossary_terms = []
    for acronym, full_def in EURO_NCAP_GLOSSARY.items():
        if re.search(rf"\b{re.escape(acronym)}\b", trimmed, re.IGNORECASE):
            glossary_terms.append(full_def)

    fallback_query = f"{trimmed} {' '.join(glossary_terms)}".strip() if glossary_terms else trimmed

    import requests
    endpoint = f"{base_url}/chat/completions"
    system_prompt = (
        "You are an expert search query optimizer for Euro NCAP vehicle testing protocols.\n"
        "Given a user question, rewrite it into a single concise search query optimized for document retrieval.\n"
        "RULES:\n"
        "1. Strictly preserve all specific acronyms (e.g. VUT, GVT, CPFA, DTLE, SOV, PBC, SLIF, BSM, ELK, LKA, iACC), numbers, units (m/s, km/h, lux, Hz, dB, Nm), and standard names (ISO, ASTM).\n"
        "2. Do NOT guess or hallucinate expansions for abbreviations you are uncertain about.\n"
        "3. Include both the acronym and key protocol nouns.\n"
        "4. Output ONLY the clean search query string with no quotes or preamble."
    )
    user_prompt = f"{glossary_prompt_text}Question: {trimmed}\nRewritten search query:"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 64,
        "temperature": 0.1
    }
    try:
        resp = requests.post(endpoint, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        rewritten = data["choices"][0]["message"]["content"].strip().strip('"').strip("'")
        return rewritten if rewritten else fallback_query
    except Exception:
        return fallback_query

def generate_llm_answer(
    query: str,
    context_str: str,
    base_url: str = config.GEN_BASE_URL,
    model_name: str = None,
    max_tokens: int = 1024
) -> str:
    """Pass user query + merged retrieved context to pretrained LLM (vLLM) for final answer."""
    if model_name is None:
        model_name = config.get_active_vllm_model(base_url)
    import requests
    endpoint = f"{base_url}/chat/completions"
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert technical specialist for Euro NCAP vehicle testing protocols.\n"
                "Answer the user's question accurately, completely, and directly based on the provided document context.\n\n"
                "CRITICAL EXTRACTION RULES:\n"
                "1. State exact numbers, speeds, tolerances, point allocations, or definitions directly.\n"
                "2. When extracting from tables, carefully align BOTH the row headers and column headers to avoid adjacent row/cell confusion.\n"
                "3. For numerical ranges, provide the full range (e.g., '0.3 to 0.6 m/s' or '0.5 to 0.7 m/s').\n"
                "4. For forbidden word lists or eligibility criteria, include all specified items.\n"
                "5. Provide no conversational filler, pleasantries, or introductory fluff."
            )
        },
        {
            "role": "user",
            "content": f"### Document Context:\n{context_str}\n\n### Question:\n{query}\n\n### Direct Answer:"
        }
    ]
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1
    }
    try:
        resp = requests.post(endpoint, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[LLM Generation Error]: Could not reach vLLM server at {base_url} ({e}). Please verify vLLM is running."


class Retriever:
    """
    Generative Retrieval Inference Engine (DSI & PAG) with Hybrid BM25 Fusion.
    Given a query, generates constrained semantic IDs and combines neural beam rankings
    with sparse lexical keyword rankings via Reciprocal Rank Fusion (RRF).
    """
    def __init__(
        self,
        run_dir: Optional[Path] = None,
        chunks_path: Path = config.CHUNKS_PATH,
        ids_path: Path = config.IDS_PATH
    ):
        if run_dir is None:
            # Check latest / available best checkpoints in priority order
            candidates = [
                config.RUNS_DIR / "dsi_smollm2-1.7b-instruct" / "best",
                config.RUNS_DIR / "dsi_qwen2.5-0.5b" / "best",
                config.RUNS_DIR / "dsi_qwen_0.5b" / "best",
            ]
            for c in candidates:
                if c.exists():
                    run_dir = c
                    break
            if run_dir is None:
                all_bests = sorted(list(config.RUNS_DIR.glob("*/best")), key=lambda p: p.stat().st_mtime, reverse=True)
                if all_bests:
                    run_dir = all_bests[0]
                else:
                    safe_name = config.MODEL_NAME.split("/")[-1].lower()
                    run_dir = config.RUNS_DIR / f"dsi_{safe_name}" / "best"

        if not run_dir.exists():
            # Fallback to last checkpoint if best not yet created
            if (run_dir.parent / "last").exists():
                run_dir = run_dir.parent / "last"
            else:
                raise FileNotFoundError(f"Model checkpoint directory not found at {run_dir}. Train model first (`python -m genret.train`).")

        print(f"Loading trained DSI model & tokenizer from: {run_dir}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(run_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            run_dir,
            torch_dtype=self.dtype,
            trust_remote_code=True
        ).to(self.device).eval()

        # Load ID structures and trie
        with open(ids_path, "r", encoding="utf-8") as f:
            ids_data = json.load(f)
        self.id_to_chunk = ids_data["id_to_chunk"]
        self.chunk_to_id = ids_data["chunk_to_id"]
        self.trie = load_trie(ids_path)

        # Load chunks store
        chunks_list = load_chunks_jsonl(chunks_path)
        self.chunks_dict = {c["chunk_id"]: c for c in chunks_list}

        # Pre-index lexical BM25 for PAG lookahead priors and hybrid fusion
        self.bm25 = BM25Retriever(chunks_path)

        # Token ID mappings for constrained decoding
        self.digit_token_ids = {i: self.tokenizer.convert_tokens_to_ids(f"<d{i}>") for i in range(10)}
        self.id_start_token_id = self.tokenizer.convert_tokens_to_ids(config.ID_START_TOKEN)
        self.id_end_token_id = self.tokenizer.convert_tokens_to_ids(config.ID_END_TOKEN)

        self.prefix_allowed_fn = make_prefix_allowed_tokens_fn(
            trie=self.trie,
            tokenizer=self.tokenizer,
            digit_token_ids=self.digit_token_ids,
            id_start_token_id=self.id_start_token_id,
            id_end_token_id=self.id_end_token_id
        )

    def retrieve(
        self, 
        query: str, 
        k: int = config.TOP_K,
        beam_width: int = config.BEAM_WIDTH,
        use_pag: bool = True,
        pag_alpha: float = getattr(config, "PAG_ALPHA", 10.0),
        early_stopping: bool = False,
        use_hybrid: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Two-Stage Generative Retrieve-then-Rerank Architecture:
        1. Stage 1 (Full-Corpus Primary Retrieval): AutoSLM Small Language Model performs
           constrained generative beam search to retrieve Top-20 candidates (Hit@20).
        2. Stage 2 (In-Candidate Reranking): Lightweight O(k) lexical alignment scores the
           20 AutoSLM-generated candidates to select the final Top-10 (k) chunks.
        """
        prompt = f"retrieve: {query}\n"
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        input_len = inputs["input_ids"].shape[1]

        # Stage 1: AutoSLM full-corpus candidate generation (Hit@20)
        candidate_pool_size = max(k * 2, 20)
        num_beams = max(beam_width, candidate_pool_size)
        num_return = min(candidate_pool_size * 2, num_beams)

        logits_processor = None
        if use_pag:
            doc_priors = self.bm25.get_all_scores(query)
            pag_proc = PAGLogitsProcessor(
                trie=self.trie,
                digit_token_ids=self.digit_token_ids,
                id_start_token_id=self.id_start_token_id,
                id_end_token_id=self.id_end_token_id,
                doc_priors=doc_priors,
                alpha=pag_alpha
            )
            logits_processor = LogitsProcessorList([pag_proc])

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.MAX_DEPTH + 3,
                num_beams=num_beams,
                num_return_sequences=num_return,
                prefix_allowed_tokens_fn=self.prefix_allowed_fn,
                logits_processor=logits_processor,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=early_stopping
            )

        sequences = outputs.sequences
        seq_scores = outputs.sequences_scores if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None else [0.0] * len(sequences)

        dsi_ranked_chunks = []
        seen_dsi_ids = set()

        for (seq, score) in zip(sequences, seq_scores):
            gen_tokens = seq[input_len:]
            gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=False)

            digits = [int(d) for d in re.findall(r"<d(\d)>", gen_text)]
            id_str = "-".join(map(str, digits))

            chunk_id = self.id_to_chunk.get(id_str)
            if chunk_id and chunk_id not in seen_dsi_ids and chunk_id in self.chunks_dict:
                seen_dsi_ids.add(chunk_id)
                chunk_obj = self.chunks_dict[chunk_id]
                raw_id = self.chunk_to_id.get(chunk_id, [])
                if isinstance(raw_id, list):
                    id_digits = raw_id
                    id_string = "-".join(map(str, id_digits))
                else:
                    id_string = str(raw_id)
                    id_digits = [int(x) for x in id_string.split("-")] if id_string else []

                dsi_ranked_chunks.append({
                    "chunk_id": chunk_id,
                    "id_path": id_digits,
                    "id_str": id_string,
                    "score": float(score),
                    "text": chunk_obj["text"]
                })
                if len(dsi_ranked_chunks) >= candidate_pool_size:
                    break

        candidate_chunks = dsi_ranked_chunks[:candidate_pool_size]
        if not candidate_chunks:
            return []

        if not use_hybrid:
            results = []
            for rank, c in enumerate(candidate_chunks[:k], start=1):
                c_copy = dict(c)
                c_copy["rank"] = rank
                results.append(c_copy)
            return results

        # Stage 2: In-Candidate Lexical Reranker (O(k) complexity across AutoSLM candidates only)
        lexical_scores = self.bm25.score_candidates(query, candidate_chunks)
        
        # Determine in-candidate lexical ranks
        sorted_by_lexical = sorted(candidate_chunks, key=lambda c: lexical_scores.get(c["chunk_id"], 0.0), reverse=True)
        in_candidate_bm25_rank = {c["chunk_id"]: rank for rank, c in enumerate(sorted_by_lexical, start=1)}
        dsi_rank_map = {c["chunk_id"]: rank for rank, c in enumerate(candidate_chunks, start=1)}

        rrf_k = 10
        w_dsi = 1.0
        w_bm25 = 1.0

        fused_scores = {}
        for c in candidate_chunks:
            cid = c["chunk_id"]
            fused_score = (w_dsi / (rrf_k + dsi_rank_map[cid])) + (w_bm25 / (rrf_k + in_candidate_bm25_rank[cid]))
            fused_scores[cid] = fused_score

        # Select Top-K from the AutoSLM candidates
        sorted_candidates = sorted(candidate_chunks, key=lambda c: fused_scores[c["chunk_id"]], reverse=True)[:k]

        results = []
        for rank, c in enumerate(sorted_candidates, start=1):
            cid = c["chunk_id"]
            c_copy = dict(c)
            c_copy["rank"] = rank
            c_copy["score"] = round(fused_scores[cid], 5)
            results.append(c_copy)

        return results

    def answer(
        self, 
        query: str, 
        top_k: int = 10, 
        max_tokens: int = 1024, 
        use_pag: bool = True,
        rewrite_query: bool = True
    ) -> Dict[str, Any]:
        """
        Unified 3-Stage Inference Pipeline:
        1. Original User Query -> vLLM Query Rewriter (Expands abbreviations & canonical keywords) -> Rewritten Search Query
        2. Rewritten Search Query -> Trained SLM (DSI + PAG) -> Semantic IDs -> Retrieve Exact Context Chunks
        3. Original User Query + Retrieved Context -> vLLM Answering -> Final Grounded Response
        """
        # Step 1: Query rewriting / expansion
        search_query = rewrite_query_for_retrieval(query) if rewrite_query else query

        # Step 2: DSI / PAG Retrieval using rewritten search query
        retrieved_chunks = self.retrieve(search_query, k=top_k, use_pag=use_pag)
        if not retrieved_chunks and search_query != query:
            # Fallback retry with raw query if rewritten returned nothing
            retrieved_chunks = self.retrieve(query, k=top_k, use_pag=use_pag)

        if not retrieved_chunks:
            return {
                "query": query,
                "search_query": search_query,
                "answer": "No matching document section found by DSI.",
                "sources": []
            }

        # Format retrieved context
        context_blocks = []
        for c in retrieved_chunks:
            context_blocks.append(f"--- [Section ID: {c['id_str']} | Chunk: {c['chunk_id']}] ---\n{c['text']}")
        context_str = "\n\n".join(context_blocks)

        sources_summary = [
            {
                "rank": c["rank"],
                "chunk_id": c["chunk_id"],
                "id_str": c["id_str"],
                "score": c["score"],
                "snippet": c["text"][:200].replace("\n", " ") + "..."
            }
            for c in retrieved_chunks
        ]

        # Step 3: LLM Answering using Original Query + Context
        llm_answer = generate_llm_answer(query=query, context_str=context_str, max_tokens=max_tokens)

        return {
            "query": query,
            "search_query": search_query,
            "answer": llm_answer,
            "sources": sources_summary
        }

def main():
    parser = argparse.ArgumentParser(description="AutoSLM DSI Retrieval + LLM Answering Pipeline")
    parser.add_argument("--query", type=str, required=True, help="Question or search query string")
    parser.add_argument("--k", type=int, default=3, help="Number of top chunks to retrieve")
    parser.add_argument("--mode", type=str, choices=["retrieve", "answer"], default="answer", help="Mode: 'retrieve' or 'answer'")
    parser.add_argument("--run_name", type=str, default=None, help="Run name under runs/")
    parser.add_argument("--no_pag", action="store_true", help="Disable PAG planning-ahead guided beam search")
    args = parser.parse_args()

    run_dir = config.RUNS_DIR / args.run_name / "best" if args.run_name else None
    retriever = Retriever(run_dir=run_dir)

    print("\n" + "=" * 65)
    print(f"QUERY: \"{args.query}\"")
    print(f"MODE: {args.mode.upper()} | PAG (Planning-Ahead): {not args.no_pag}")
    print("=" * 65)

    if args.mode == "retrieve":
        results = retriever.retrieve(args.query, k=args.k, use_pag=not args.no_pag)
        for res in results:
            print(f"\n#{res['rank']} [Score: {res['score']:.4f}] Chunk ID: {res['chunk_id']} (Semantic ID: {res['id_str']})")
            snippet = res['text'][:250].replace("\n", " ")
            print(f"Text: \"{snippet}...\"")
    else:
        qa_result = retriever.answer(args.query, top_k=args.k, use_pag=not args.no_pag)
        print(f"\n💡 ANSWER:\n{qa_result['answer']}\n")
        print("-" * 65)
        print("📚 RETRIEVED GROUNDING SOURCES (Trained DSI Model):")
        for src in qa_result["sources"]:
            print(f"  • Rank #{src['rank']} | Chunk {src['chunk_id']} [ID: {src['id_str']}] (Score: {src['score']:.4f})")
            print(f"    \"{src['snippet']}\"")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
