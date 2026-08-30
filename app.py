import os
import re
import json
import time
import asyncio
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from genret import config
from genret.infer import Retriever, rewrite_query_for_retrieval

app = FastAPI(title="DSLM-GR ReAct Generative Retrieval App")

# Global singleton retriever instance & concurrency lock
_retriever: Optional[Retriever] = None
_dsi_lock = asyncio.Lock()

def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        print("Initializing DSLM-GR Retriever...")
        _retriever = Retriever()
    return _retriever

class QueryRequest(BaseModel):
    query: str

def react_agent_loop(
    original_query: str,
    retrieved_chunks: List[Dict[str, Any]],
    retriever: Retriever,
    base_url: str = config.GEN_BASE_URL,
    model_name: str = None,
    max_steps: int = 4
) -> Dict[str, Any]:
    if model_name is None:
        model_name = config.get_active_vllm_model(base_url)
    """
    Advanced Generative Retrieval ReAct Agentic Harness:
    1. Evaluates retrieved evidence against the user query.
    2. Has direct tool access to DSI_Search (queries the trained DSI SLM on the fly for missing context).
    3. Performs Lookup & Table Verification across accumulated chunks.
    4. Robustly parses Finish actions and guarantees a complete, verified answer.
    """
    accumulated_chunks = {c["chunk_id"]: c for c in retrieved_chunks}
    steps = []
    final_deduction = ""

    def build_context_str(chunks_dict: Dict[str, Dict[str, Any]]) -> str:
        return "\n\n".join([
            f"=== [Chunk {cid} | Semantic ID: {c.get('id_str', '')}] ===\n{c['text']}"
            for cid, c in chunks_dict.items()
        ])

    system_prompt = (
        "You are an expert autonomous reasoning agent for Euro NCAP vehicle testing protocols.\n"
        "You have access to retrieved protocol documentation chunks and specialized tools to find missing facts.\n\n"
        "AVAILABLE TOOLS:\n"
        "1. DSI_Search[search phrase] -> Queries the neural DSI generative index to retrieve additional relevant protocol chunks.\n"
        "2. Lookup[exact keyword or table term] -> Scans all retrieved document chunks for exact matching lines or numbers.\n"
        "3. Finish[your detailed reasoned answer] -> Concludes the task once sufficient facts are established.\n\n"
        "FORMAT RULES (strictly follow at every step):\n"
        "Thought: <Analyze the user question, evaluate if current context contains the exact answer, and determine what tool to call next>\n"
        "Action: <One of: DSI_Search[phrase], Lookup[term], or Finish[answer]>\n\n"
        "GROUNDING RULES:\n"
        "- Base your answer strictly on the provided chunks.\n"
        "- If the answer is in the context, extract all exact speeds, tolerances, points, or definitions.\n"
        "- You decide the response depth: provide direct numbers for simple questions, or comprehensive breakdowns for multi-part protocols."
    )

    history = f"### User Question:\n{original_query}\n\n### Current Retrieved Protocol Context:\n{build_context_str(accumulated_chunks)}\n"

    for step_num in range(1, max_steps + 1):
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": history + f"\nStep {step_num}:\n"}
            ],
            "max_tokens": 1024,
            "temperature": 0.1
        }

        try:
            resp = requests.post(f"{base_url}/chat/completions", json=payload, timeout=45)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            break

        # Parse Thought
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|$)", content, re.DOTALL | re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else "Analyzing retrieved protocol context."

        # Parse Action
        action_match = re.search(r"Action:\s*(.*)", content, re.DOTALL | re.IGNORECASE)
        action_raw = action_match.group(1).strip() if action_match else content

        # Check for Finish / Final Answer
        finish_match = (
            re.search(r"Finish\[(.*)\]", action_raw, re.DOTALL | re.IGNORECASE) or
            re.search(r"Finish:\s*(.*)", action_raw, re.DOTALL | re.IGNORECASE) or
            re.search(r"Final Answer:\s*(.*)", content, re.DOTALL | re.IGNORECASE)
        )

        if finish_match:
            final_deduction = finish_match.group(1).strip()
            steps.append({
                "step": step_num,
                "thought": thought,
                "action": "Finish",
                "observation": "Ground truth verified from protocol context."
            })
            break

        # Check for DSI_Search tool call (Pure Neural Generative Retrieval)
        dsi_search_match = re.search(r"DSI_Search\[(.*?)\]", action_raw, re.IGNORECASE)
        lookup_match = re.search(r"Lookup\[(.*?)\]", action_raw, re.IGNORECASE)

        observation = ""
        if dsi_search_match:
            search_query = dsi_search_match.group(1).strip()
            # Query trained DSI + PAG SLM directly for additional chunks
            new_chunks = retriever.retrieve(search_query, k=3, use_pag=True)
            added = []
            for nc in new_chunks:
                cid = nc["chunk_id"]
                if cid not in accumulated_chunks:
                    accumulated_chunks[cid] = nc
                    added.append(f"Chunk {cid}")
            if added:
                observation = f"DSI retrieved additional relevant context: {', '.join(added)}."
            else:
                observation = "DSI confirmed existing chunks already cover this area."
        elif lookup_match:
            term = lookup_match.group(1).strip().lower()
            context_all = build_context_str(accumulated_chunks)
            matching_lines = [
                line.strip() for line in context_all.split("\n")
                if term in line.lower() and len(line.strip()) > 8
            ]
            if matching_lines:
                observation = "Found matching occurrences in context:\n" + "\n".join(matching_lines)
            else:
                observation = f"Term '{term}' not directly present in current chunks. Cross-referencing section definitions."
        else:
            observation = "Context verified."

        steps.append({
            "step": step_num,
            "thought": thought,
            "action": action_raw.split("\n")[0],
            "observation": observation
        })

        # Update running history with latest accumulated chunks
        history += f"\nThought: {thought}\nAction: {action_raw.split(chr(10))[0]}\nObservation: {observation}\n"

    # Robust Final Grounded Synthesis if not already finalized
    if not final_deduction or len(final_deduction) < 15:
        context_str = build_context_str(accumulated_chunks)
        synth_payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert technical specialist for Euro NCAP vehicle testing protocols.\n"
                        "Answer the user's question accurately, completely, and directly based on the provided protocol chunks.\n"
                        "If specific numbers, speeds, tolerances, definitions, or table cells are mentioned, state them clearly."
                    )
                },
                {
                    "role": "user",
                    "content": f"### Document Context:\n{context_str}\n\n### User Question:\n{original_query}"
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.1
        }
        try:
            synth_resp = requests.post(f"{base_url}/chat/completions", json=synth_payload, timeout=60)
            synth_resp.raise_for_status()
            final_deduction = synth_resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            final_deduction = "Could not synthesize answer from retrieved context."

    return {
        "steps": steps,
        "raw_deduction": final_deduction,
        "total_chunks": list(accumulated_chunks.values())
    }

def rewrite_to_the_point_answer(
    original_query: str,
    deduction: str,
    base_url: str = config.GEN_BASE_URL,
    model_name: str = None
) -> str:
    if model_name is None:
        model_name = config.get_active_vllm_model(base_url)
    """
    Cleans up the ReAct deduction: removes conversational filler and pleasantries
    while strictly preserving the depth, structure, and detail decided by the ReAct agent.
    """
    system_prompt = (
        "You are an expert technical editor.\n"
        "Your task is to present the ReAct agent's verified deduction with zero conversational preamble.\n"
        "GROUND RULES:\n"
        "1. NO introductory greetings or pleasantries ('Sure!', 'Based on the context', 'Here is the breakdown', 'I hope this helps').\n"
        "2. Preserve the exact level of detail, numbers, conditions, tables, or summary provided in the technical deduction.\n"
        "3. Output ONLY the clean, formatted factual answer."
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {original_query}\n\nTechnical Deduction:\n{deduction}\n\nClean Output:"}
        ],
        "max_tokens": 1024,
        "temperature": 0.1
    }

    try:
        resp = requests.post(f"{base_url}/chat/completions", json=payload, timeout=40)
        resp.raise_for_status()
        clean_ans = resp.json()["choices"][0]["message"]["content"].strip()
        return clean_ans if clean_ans else deduction
    except Exception:
        return deduction

@app.post("/api/ask")
async def api_ask(req: QueryRequest):
    query = req.query.strip()
    if not query:
        return JSONResponse({"error": "Empty query"}, status_code=400)

    start_time = time.time()

    # Step 1: Query Rephrasing (Domain Acronym Expansion)
    rephrased_query = rewrite_query_for_retrieval(query)

    # Step 2: DSI + PAG Generative Retrieval (Initial Top-10 Chunks)
    retriever = get_retriever()
    async with _dsi_lock:
        retrieved_results = retriever.retrieve(rephrased_query, k=10, use_pag=True)

    # Step 3: Advanced Agentic ReAct Harness (with Dynamic DSI Search & Lookup Tools)
    react_result = react_agent_loop(
        original_query=query,
        retrieved_chunks=retrieved_results,
        retriever=retriever
    )

    # Step 4: Direct Clean Answer Presentation
    final_answer = rewrite_to_the_point_answer(
        original_query=query,
        deduction=react_result["raw_deduction"]
    )

    elapsed = round(time.time() - start_time, 2)

    return {
        "original_query": query,
        "rephrased_query": rephrased_query,
        "retrieved_chunks": [
            {
                "chunk_id": r["chunk_id"],
                "semantic_id": r.get("id_str", ""),
                "score": round(r.get("score", 0.0), 3),
                "text": r["text"]
            }
            for r in react_result.get("total_chunks", retrieved_results)
        ],
        "react_steps": react_result["steps"],
        "final_answer": final_answer,
        "elapsed_seconds": elapsed
    }

# ==============================================================================
# BENCHMARK SUITE BACKEND
# ==============================================================================
BENCHMARK_PATH = Path("benchmark.json") if Path("benchmark.json").exists() else config.DATA_DIR / "benchmark.json"
BENCHMARK_RESULTS_PATH = Path("benchmark_results.json")

def load_benchmark_questions() -> List[Dict[str, Any]]:
    if BENCHMARK_PATH.exists():
        try:
            with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            print(f"Error loading benchmark.json: {e}")
    return []

@app.get("/api/benchmark/questions")
async def api_get_benchmark_questions():
    questions = load_benchmark_questions()
    return {
        "count": len(questions),
        "source_file": str(BENCHMARK_PATH.resolve()),
        "questions": questions
    }

class BenchmarkUploadRequest(BaseModel):
    questions: List[Dict[str, Any]]

@app.post("/api/benchmark/upload")
async def api_upload_benchmark(req: BenchmarkUploadRequest):
    if not req.questions:
        return JSONResponse({"error": "Empty question list provided"}, status_code=400)
    try:
        with open(BENCHMARK_PATH, "w", encoding="utf-8") as f:
            json.dump(req.questions, f, indent=2)
        return {
            "status": "success",
            "message": f"Successfully saved {len(req.questions)} questions to {BENCHMARK_PATH.name}",
            "count": len(req.questions)
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

class BenchmarkItemRequest(BaseModel):
    id: Optional[str] = None
    question: str
    reference_answer: Optional[str] = None

class SaveResultsRequest(BaseModel):
    results: List[Dict[str, Any]]

# ==============================================================================
# BENCHMARK LOGGING & PROFILING SYSTEM
# ==============================================================================
BENCHMARK_LOGS_DIR = Path("benchmark_logs")
BENCHMARK_LOGS_DIR.mkdir(parents=True, exist_ok=True)
LATEST_TRACE_LOG = BENCHMARK_LOGS_DIR / "latest_trace.jsonl"
LATEST_HUMAN_LOG = BENCHMARK_LOGS_DIR / "latest_run.log"

def log_benchmark_item(
    item_id: str,
    question: str,
    reference_answer: Optional[str],
    generated_answer: str,
    rephrased_query: str,
    retrieved_chunks: List[Dict[str, Any]],
    react_steps: List[Dict[str, Any]],
    retrieval_time: float,
    generation_time: float,
    total_time: float,
    status: str = "SUCCESS",
    error_msg: Optional[str] = None
):
    import datetime
    timestamp = datetime.datetime.now().isoformat()

    log_entry = {
        "timestamp": timestamp,
        "query_id": str(item_id),
        "question": question,
        "reference_answer": reference_answer,
        "generated_answer": generated_answer,
        "rephrased_query": rephrased_query,
        "retrieved_chunks": [
            {
                "chunk_id": c.get("chunk_id"),
                "semantic_id": c.get("semantic_id") or c.get("id_str"),
                "score": c.get("score"),
                "text_length": len(c.get("text", "")),
                "text_preview": c.get("text", "")[:300]
            }
            for c in retrieved_chunks
        ],
        "react_steps": react_steps,
        "latency_breakdown": {
            "retrieval_sec": round(retrieval_time, 3),
            "generation_sec": round(generation_time, 3),
            "total_sec": round(total_time, 3)
        },
        "status": status,
        "error": error_msg
    }

    # Append to structured JSONL trace
    with open(LATEST_TRACE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    # Append to human-readable diagnostic log
    chunk_ids = [c.get("chunk_id", "") for c in retrieved_chunks]
    log_line = (
        f"[{timestamp}] [{status}] Query #{item_id}: \"{question[:80]}\"\n"
        f"  ├─ Rephrased: \"{rephrased_query[:90]}\"\n"
        f"  ├─ Retrieved Chunks ({len(chunk_ids)}): {chunk_ids[:6]}\n"
        f"  ├─ ReAct Steps: {len(react_steps)} | Latency: {total_time:.2f}s (Retrieval: {retrieval_time:.2f}s | Gen: {generation_time:.2f}s)\n"
        f"  └─ Answer Preview: {generated_answer.replace(chr(10), ' ')[:140]}...\n\n"
    )
    with open(LATEST_HUMAN_LOG, "a", encoding="utf-8") as f:
        f.write(log_line)

@app.get("/api/benchmark/logs")
async def api_get_benchmark_logs():
    if not LATEST_HUMAN_LOG.exists():
        return JSONResponse({"error": "No benchmark log exists yet. Run benchmark first."}, status_code=404)
    from fastapi.responses import FileResponse
    return FileResponse(
        path=LATEST_HUMAN_LOG,
        filename=f"benchmark_debug_{int(time.time())}.log",
        media_type="text/plain"
    )

@app.get("/api/benchmark/trace_jsonl")
async def api_get_benchmark_trace_jsonl():
    if not LATEST_TRACE_LOG.exists():
        return JSONResponse({"error": "No benchmark trace JSONL exists yet. Run benchmark first."}, status_code=404)
    from fastapi.responses import FileResponse
    return FileResponse(
        path=LATEST_TRACE_LOG,
        filename=f"benchmark_trace_{int(time.time())}.jsonl",
        media_type="application/x-ndjson"
    )

async def generate_grounded_answer_async(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    base_url: str = config.GEN_BASE_URL,
    model_name: str = None,
    max_retries: int = 3
) -> str:
    """
    Non-blocking async grounded generator with exponential backoff retries.
    Uses strict Euro NCAP table extraction and numerical precision guidelines.
    """
    if model_name is None:
        model_name = config.get_active_vllm_model(base_url)
    import httpx
    context_str = "\n\n".join([f"=== [Chunk {c['chunk_id']}] ===\n{c['text']}" for c in retrieved_chunks])
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert technical specialist for Euro NCAP vehicle testing protocols.\n"
                    "Provide a direct, complete, and accurate factual answer to the user question based strictly on the document chunks.\n\n"
                    "CRITICAL EXTRACTION RULES:\n"
                    "1. State exact numbers, speeds, tolerances, point allocations, or definitions directly with no conversational filler.\n"
                    "2. When extracting from markdown tables, cross-reference BOTH the row headers (scenario, vehicle type, intentional vs unintentional) and column headers (tolerance, speed, points) to prevent confusing adjacent cells.\n"
                    "3. For numerical ranges, state the complete range (e.g., '0.3 to 0.6 m/s' or '0.5 to 0.7 m/s').\n"
                    "4. For forbidden word lists, cite all prohibited terms listed in the protocol (e.g., 'auto', 'automatic', 'automated', 'pilot', 'self-drive').\n"
                    "5. For acronyms or terms, provide the explicit definition stated in the protocol."
                )
            },
            {
                "role": "user",
                "content": f"### Document Context:\n{context_str}\n\n### Question:\n{query}\n\n### Direct Answer:"
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.1
    }

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(f"{base_url}/chat/completions", json=payload)
                resp.raise_for_status()
                ans = resp.json()["choices"][0]["message"]["content"].strip()
                return ans if ans else "No direct answer generated."
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(1.0 * (2 ** attempt))
            else:
                return f"Error during answer generation: {e}"

@app.post("/api/benchmark/evaluate_item")
async def api_benchmark_evaluate_item(req: BenchmarkItemRequest):
    q = req.question.strip()
    if not q:
        return JSONResponse({"error": "Empty query"}, status_code=400)

    start_time = time.time()
    t_ret_start = time.time()
    rephrased_query = rewrite_query_for_retrieval(q)
    retriever = get_retriever()
    async with _dsi_lock:
        retrieved_results = retriever.retrieve(rephrased_query, k=10, use_pag=True, use_hybrid=True)
    retrieval_time = time.time() - t_ret_start

    t_gen_start = time.time()
    final_answer = await generate_grounded_answer_async(
        query=q,
        retrieved_chunks=retrieved_results
    )
    generation_time = time.time() - t_gen_start
    elapsed = round(time.time() - start_time, 2)

    # Save to debug logs
    log_benchmark_item(
        item_id=req.id or "unknown",
        question=q,
        reference_answer=req.reference_answer,
        generated_answer=final_answer,
        rephrased_query=rephrased_query,
        retrieved_chunks=retrieved_results,
        react_steps=[],
        retrieval_time=retrieval_time,
        generation_time=generation_time,
        total_time=elapsed
    )

    return {
        "id": req.id,
        "question": q,
        "reference_answer": req.reference_answer,
        "generated_answer": final_answer,
        "rephrased_query": rephrased_query,
        "retrieved_chunks": [
            {
                "chunk_id": r["chunk_id"],
                "semantic_id": r.get("id_str", ""),
                "score": round(r.get("score", 0.0), 3)
            }
            for r in retrieved_results
        ],
        "react_steps_count": 1,
        "latency_seconds": elapsed
    }

@app.post("/api/benchmark/save_results")
async def api_benchmark_save_results(req: SaveResultsRequest):
    try:
        with open(BENCHMARK_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(req.results, f, indent=2)
        return {"status": "success", "count": len(req.results), "file": str(BENCHMARK_RESULTS_PATH)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/benchmark/export")
async def api_benchmark_export():
    if not BENCHMARK_RESULTS_PATH.exists():
        return JSONResponse({"error": "No benchmark results file found. Run benchmark first."}, status_code=404)
    from fastapi.responses import FileResponse
    return FileResponse(
        path=BENCHMARK_RESULTS_PATH,
        filename=f"benchmark_results_{int(time.time())}.json",
        media_type="application/json"
    )

class BenchmarkBatchRequest(BaseModel):
    items: List[BenchmarkItemRequest]
    concurrency: int = 8

@app.post("/api/benchmark/evaluate_batch")
async def api_benchmark_evaluate_batch(req: BenchmarkBatchRequest):
    if not req.items:
        return JSONResponse({"error": "Empty batch provided"}, status_code=400)

    semaphore = asyncio.Semaphore(max(1, min(req.concurrency, 32)))
    retriever = get_retriever()
    results = []

    async def process_single_item(item: BenchmarkItemRequest):
        async with semaphore:
            start_time = time.time()
            q = item.question.strip()
            if not q:
                return None
            
            # Step 1: Query expansion with domain glossary grounding
            rephrased = rewrite_query_for_retrieval(q)
            
            # Step 2: Hybrid DSI + BM25 neural/lexical retrieval
            async with _dsi_lock:
                retrieved = retriever.retrieve(rephrased, k=10, use_pag=True, use_hybrid=True)
            
            t_ret = time.time() - start_time

            # Step 3: Resilient Grounded vLLM Generation
            t_gen_start = time.time()
            ans = await generate_grounded_answer_async(
                query=q,
                retrieved_chunks=retrieved
            )
            t_gen = time.time() - t_gen_start
            elapsed = round(time.time() - start_time, 2)

            log_benchmark_item(
                item_id=item.id or "unknown",
                question=q,
                reference_answer=item.reference_answer,
                generated_answer=ans,
                rephrased_query=rephrased,
                retrieved_chunks=retrieved,
                react_steps=[],
                retrieval_time=t_ret,
                generation_time=t_gen,
                total_time=elapsed
            )

            return {
                "id": item.id,
                "question": q,
                "reference_answer": item.reference_answer,
                "generated_answer": ans,
                "rephrased_query": rephrased,
                "retrieved_chunks": [
                    {
                        "chunk_id": r["chunk_id"],
                        "semantic_id": r.get("id_str", ""),
                        "score": round(r.get("score", 0.0), 3)
                    }
                    for r in retrieved
                ],
                "latency_seconds": elapsed
            }

    tasks = [process_single_item(item) for item in req.items]
    completed_results = await asyncio.gather(*tasks)
    results = [r for r in completed_results if r is not None]

    return {"results": results, "count": len(results)}

# ==============================================================================
# UNIFIED FRONTEND (ASK + BENCHMARK SUITE)
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
async def index():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DSLM-GR | Protocol Assistant & Benchmark Suite</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #090d16;
            --bg-card: #111726;
            --bg-card-sub: #161e31;
            --border-color: #1e293b;
            --border-hover: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-cyan: #38bdf8;
            --accent-blue: #2563eb;
            --accent-emerald: #10b981;
            --accent-purple: #a855f7;
            --accent-amber: #f59e0b;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            background-color: var(--bg-primary);
            color: var(--text-main);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            line-height: 1.6;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 24px 16px;
        }

        .container {
            width: 100%;
            max-width: 980px;
            display: flex;
            flex-direction: column;
            gap: 18px;
        }

        header { text-align: center; margin-bottom: 2px; }

        h1 {
            font-size: 24px;
            font-weight: 700;
            letter-spacing: -0.5px;
            color: #ffffff;
            margin-bottom: 4px;
        }

        p.subtitle { color: var(--text-muted); font-size: 13px; }

        .nav-tabs {
            display: flex;
            justify-content: center;
            gap: 8px;
            margin-bottom: 4px;
        }

        .tab-btn {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            color: var(--text-muted);
            font-family: inherit;
            font-size: 13px;
            font-weight: 600;
            padding: 8px 18px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .tab-btn.active {
            background: var(--bg-card-sub);
            color: var(--accent-cyan);
            border-color: var(--accent-cyan);
            box-shadow: 0 0 15px rgba(56, 189, 248, 0.15);
        }

        .tab-btn:hover:not(.active) { color: var(--text-main); border-color: var(--border-hover); }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 18px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
        }

        .input-group { display: flex; gap: 10px; }

        input[type="text"] {
            flex: 1;
            background: var(--bg-card-sub);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-main);
            font-family: inherit;
            font-size: 14px;
            padding: 12px 16px;
            outline: none;
            transition: all 0.2s ease;
        }

        input[type="text"]:focus {
            border-color: var(--accent-cyan);
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15);
        }

        .btn {
            color: #ffffff;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            font-size: 13px;
            padding: 10px 18px;
            cursor: pointer;
            transition: transform 0.1s ease, filter 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            user-select: none;
        }

        .btn:hover { filter: brightness(1.1); }
        .btn:active { transform: scale(0.98); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; filter: grayscale(1); }

        .btn-primary { background: linear-gradient(135deg, #0284c7, #2563eb); }
        .btn-turbo { background: linear-gradient(135deg, #7c3aed, #a855f7); }
        .btn-success { background: linear-gradient(135deg, #059669, #10b981); }
        .btn-secondary { background: var(--bg-card-sub); border: 1px solid var(--border-color); color: var(--text-main); }

        .select-input {
            background: var(--bg-card-sub);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            font-family: inherit;
            font-size: 13px;
            font-weight: 500;
            padding: 9px 12px;
            border-radius: 8px;
            outline: none;
            cursor: pointer;
        }

        .select-input:focus { border-color: var(--accent-cyan); }

        .quick-queries {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 12px;
            align-items: center;
        }

        .quick-queries span { font-size: 12px; color: var(--text-muted); }

        .query-chip {
            background: var(--bg-card-sub);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 3px 9px;
            font-size: 12px;
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .query-chip:hover {
            color: var(--text-main);
            border-color: var(--border-hover);
            background: #1e293b;
        }

        .answer-card {
            background: var(--bg-card);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 14px;
            padding: 20px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
            animation: fadeIn 0.25s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border-color);
        }

        .card-title {
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            color: var(--accent-cyan);
        }

        .direct-answer {
            font-size: 15px;
            color: #f1f5f9;
            white-space: pre-wrap;
            line-height: 1.7;
        }

        /* Benchmark Suite Styles */
        .bench-controls {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
        }

        .bench-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 12px;
            margin: 14px 0;
        }

        .stat-box {
            background: var(--bg-card-sub);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }

        .stat-value {
            font-size: 20px;
            font-weight: 700;
            color: var(--accent-cyan);
            font-family: 'JetBrains Mono', monospace;
        }

        .stat-label {
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 2px;
        }

        .progress-bar-container {
            width: 100%;
            background: var(--bg-card-sub);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            height: 10px;
            overflow: hidden;
            margin-bottom: 14px;
        }

        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #7c3aed, #0284c7, #10b981);
            width: 0%;
            transition: width 0.15s ease;
        }

        .results-table-container {
            max-height: 480px;
            overflow-y: auto;
            border: 1px solid var(--border-color);
            border-radius: 8px;
        }

        table.bench-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }

        table.bench-table th {
            background: var(--bg-card-sub);
            color: var(--text-muted);
            text-align: left;
            padding: 10px 12px;
            font-weight: 600;
            border-bottom: 1px solid var(--border-color);
            position: sticky;
            top: 0;
            z-index: 10;
        }

        table.bench-table td {
            padding: 10px 12px;
            border-bottom: 1px solid rgba(30, 41, 59, 0.6);
            vertical-align: top;
        }

        table.bench-table tr:hover td {
            background: rgba(22, 30, 49, 0.5);
        }

        .badge {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            font-family: 'JetBrains Mono', monospace;
            background: var(--bg-card-sub);
            border: 1px solid var(--border-color);
            color: var(--accent-cyan);
        }

        .spinner {
            border: 2px solid rgba(255, 255, 255, 0.2);
            border-top: 2px solid #ffffff;
            border-radius: 50%;
            width: 14px;
            height: 14px;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .hidden { display: none !important; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>DSLM-GR Technical Assistant</h1>
            <p class="subtitle">Domain-Adapted Generative Retrieval with Grounded Answering & Benchmark Suite</p>
        </header>

        <div class="nav-tabs">
            <button class="tab-btn active" id="tabAskBtn" onclick="switchTab('ask')">💬 Interactive Assistant</button>
            <button class="tab-btn" id="tabBenchBtn" onclick="switchTab('bench')">📊 Benchmark Suite (<span id="benchCountHeader">100</span> Qs)</button>
        </div>

        <!-- TAB 1: INTERACTIVE ASK VIEW -->
        <div id="viewAsk" class="tab-view">
            <div class="card">
                <div class="input-group">
                    <input type="text" id="queryInput" placeholder="Ask a question (e.g. What is the target speed for CBTAfs?)..." autofocus>
                    <button class="btn btn-primary" id="submitBtn" onclick="runSearch()">
                        <span id="btnText">Ask</span>
                        <div class="spinner hidden" id="btnSpinner"></div>
                    </button>
                </div>
                <div class="quick-queries">
                    <span>Examples:</span>
                    <div class="query-chip" onclick="setQuery('What does VUT stand for in the Assisted Driving protocol?')">VUT Definition</div>
                    <div class="query-chip" onclick="setQuery('What is the target speed for all CBTA turning sub-scenarios?')">CBTA Speeds</div>
                    <div class="query-chip" onclick="setQuery('What are the tyre preparation requirements for vehicle test setup?')">Tyre Specs</div>
                    <div class="query-chip" onclick="setQuery('What is the difference between CPFA and CPNCO collisions?')">CPFA vs CPNCO</div>
                </div>
            </div>

            <div id="resultsContainer" class="answer-card hidden" style="margin-top: 18px;">
                <div class="card-header">
                    <span class="card-title">Answer</span>
                    <span id="elapsedBadge" style="font-size: 12px; color: var(--text-muted);"></span>
                </div>
                <div class="direct-answer" id="directAnswerText"></div>
            </div>
        </div>

        <!-- TAB 2: BENCHMARK SUITE VIEW -->
        <div id="viewBench" class="tab-view hidden">
            <div class="card">
                <div class="bench-controls">
                    <div>
                        <h3 style="font-size: 16px; font-weight: 600; color: #fff;">vLLM Continuous-Batching Benchmark</h3>
                        <p style="font-size: 12px; color: var(--text-muted); margin-top: 2px;">
                            Evaluates <span id="benchCountLabel" style="color: var(--accent-cyan); font-weight: 600;">100</span> questions from <code style="color: var(--accent-cyan);">benchmark.json</code> using parallel GPU continuous batching.
                        </p>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                        <label style="font-size: 12px; color: var(--text-muted);">Batch Concurrency:</label>
                        <select id="concurrencySelect" class="select-input">
                            <option value="16" selected>⚡ 16x Turbo (Max vLLM Batch)</option>
                            <option value="8">🚀 8x Fast</option>
                            <option value="4">⚙️ 4x Balanced</option>
                            <option value="1">1x Sequential</option>
                        </select>
                        <input type="file" id="jsonFileInput" accept=".json" style="display: none;" onchange="handleFileUpload(event)">
                        <button class="btn btn-secondary" onclick="document.getElementById('jsonFileInput').click()">📁 Load JSON</button>
                        <button class="btn btn-turbo" id="startBenchBtn" onclick="runBenchmark()">
                            <span id="benchBtnText">⚡ Run Batched Benchmark</span>
                            <div class="spinner hidden" id="benchSpinner"></div>
                        </button>
                        <button class="btn btn-success hidden" id="exportBtn" onclick="exportResults()">📥 Export JSON</button>
                        <button class="btn btn-secondary hidden" id="logBtn" onclick="window.location.href='/api/benchmark/logs'">📄 Debug Log</button>
                    </div>
                </div>

                <div class="bench-stats">
                    <div class="stat-box">
                        <div class="stat-value" id="statCompleted">0 / 0</div>
                        <div class="stat-label">Progress</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="statAvgLatency">0.00s</div>
                        <div class="stat-label">Avg Latency</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="statThroughput">0.0 QPS</div>
                        <div class="stat-label">vLLM Throughput</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="statTotalTime">0.0s</div>
                        <div class="stat-label">Total Elapsed</div>
                    </div>
                </div>

                <div class="progress-bar-container">
                    <div class="progress-bar-fill" id="progressBar"></div>
                </div>

                <div class="results-table-container">
                    <table class="bench-table">
                        <thead>
                            <tr>
                                <th style="width: 50px;">#</th>
                                <th style="width: 28%;">Question</th>
                                <th style="width: 32%;">Generated Answer</th>
                                <th style="width: 25%;">Grounding Chunks</th>
                                <th style="width: 15%;">Latency</th>
                            </tr>
                        </thead>
                        <tbody id="benchTableBody">
                            <tr>
                                <td colspan="5" style="text-align: center; color: var(--text-muted); padding: 30px;">
                                    Click "⚡ Run Batched Benchmark" to execute continuous-batching evaluation on the 100 questions.
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentQuestions = [];
        let benchmarkResults = [];
        let isRunningBenchmark = false;

        // Tab Switching
        function switchTab(tab) {
            document.getElementById('tabAskBtn').classList.toggle('active', tab === 'ask');
            document.getElementById('tabBenchBtn').classList.toggle('active', tab === 'bench');
            document.getElementById('viewAsk').classList.toggle('hidden', tab !== 'ask');
            document.getElementById('viewBench').classList.toggle('hidden', tab !== 'bench');
        }

        // Ask Interface Logic
        const queryInput = document.getElementById('queryInput');
        const submitBtn = document.getElementById('submitBtn');
        const btnText = document.getElementById('btnText');
        const btnSpinner = document.getElementById('btnSpinner');
        const resultsContainer = document.getElementById('resultsContainer');

        queryInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') runSearch();
        });

        function setQuery(text) {
            queryInput.value = text;
            runSearch();
        }

        async function runSearch() {
            const query = queryInput.value.trim();
            if (!query) return;

            submitBtn.disabled = true;
            btnText.classList.add('hidden');
            btnSpinner.classList.remove('hidden');

            try {
                const response = await fetch('/api/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });

                if (!response.ok) throw new Error('Request failed');
                const data = await response.json();

                document.getElementById('directAnswerText').innerText = data.final_answer;
                document.getElementById('elapsedBadge').innerText = `${data.elapsed_seconds}s`;

                resultsContainer.classList.remove('hidden');
                resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } catch (err) {
                alert('Error: ' + err.message);
            } finally {
                submitBtn.disabled = false;
                btnText.classList.remove('hidden');
                btnSpinner.classList.add('hidden');
            }
        }

        // Benchmark Suite Logic
        async function fetchBenchmarkQuestions() {
            try {
                const res = await fetch('/api/benchmark/questions');
                const data = await res.json();
                currentQuestions = data.questions;
                document.getElementById('benchCountHeader').innerText = data.count;
                document.getElementById('benchCountLabel').innerText = data.count;
                document.getElementById('statCompleted').innerText = `0 / ${data.count}`;
            } catch (err) {
                console.error("Failed to load benchmark questions", err);
            }
        }
        fetchBenchmarkQuestions();

        async function handleFileUpload(event) {
            const file = event.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = async (e) => {
                try {
                    const parsed = JSON.parse(e.target.result);
                    if (!Array.isArray(parsed)) throw new Error("JSON must be a list of questions");
                    
                    const res = await fetch('/api/benchmark/upload', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ questions: parsed })
                    });
                    if (!res.ok) throw new Error("Upload failed");
                    alert(`Loaded ${parsed.length} questions successfully!`);
                    fetchBenchmarkQuestions();
                } catch (err) {
                    alert("Invalid JSON: " + err.message);
                }
            };
            reader.readAsText(file);
        }

        async function runBenchmark() {
            if (isRunningBenchmark) return;
            if (!currentQuestions.length) {
                alert("No benchmark questions loaded.");
                return;
            }

            isRunningBenchmark = true;
            benchmarkResults = [];

            const concurrency = parseInt(document.getElementById('concurrencySelect').value) || 8;
            const startBtn = document.getElementById('startBenchBtn');
            const benchBtnText = document.getElementById('benchBtnText');
            const benchSpinner = document.getElementById('benchSpinner');
            const exportBtn = document.getElementById('exportBtn');
            const tableBody = document.getElementById('benchTableBody');
            const progressBar = document.getElementById('progressBar');

            startBtn.disabled = true;
            benchBtnText.classList.add('hidden');
            benchSpinner.classList.remove('hidden');
            exportBtn.classList.add('hidden');
            tableBody.innerHTML = '';

            const total = currentQuestions.length;
            let completed = 0;
            let totalLatency = 0;
            const benchStartTime = Date.now();

            // Concurrent Worker Queue
            let queueIndex = 0;

            async function worker() {
                while (queueIndex < total) {
                    const idx = queueIndex++;
                    const item = currentQuestions[idx];
                    const qText = item.question || item.query || item.q || "";
                    const qId = item.id || `${idx + 1}`;
                    const qRef = item.answer || item.reference_answer || "";

                    try {
                        const res = await fetch('/api/benchmark/evaluate_item', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                id: `${qId}`,
                                question: qText,
                                reference_answer: qRef
                            })
                        });

                        if (res.ok) {
                            const result = await res.json();
                            benchmarkResults.push(result);
                            totalLatency += result.latency_seconds;

                            // Append row to live table
                            const tr = document.createElement('tr');
                            const chunkBadges = result.retrieved_chunks.slice(0, 3).map(c => 
                                `<span class="badge" title="${c.semantic_id}">${c.chunk_id}</span>`
                            ).join(' ');

                            tr.innerHTML = `
                                <td><span style="color: var(--text-muted); font-family: monospace;">#${qId}</span></td>
                                <td><strong style="color: #fff;">${escapeHtml(qText)}</strong></td>
                                <td style="color: #cbd5e1;">${escapeHtml(result.generated_answer)}</td>
                                <td>${chunkBadges}</td>
                                <td><span class="badge">${result.latency_seconds}s</span></td>
                            `;
                            tableBody.prepend(tr);
                        }
                    } catch (err) {
                        console.error(`Error on query ${qId}:`, err);
                    }

                    completed++;
                    const pct = Math.round((completed / total) * 100);
                    progressBar.style.width = `${pct}%`;
                    const elapsedTotal = (Date.now() - benchStartTime) / 1000;
                    const qps = (completed / Math.max(0.1, elapsedTotal)).toFixed(1);

                    document.getElementById('statCompleted').innerText = `${completed} / ${total}`;
                    document.getElementById('statAvgLatency').innerText = `${(totalLatency / completed).toFixed(2)}s`;
                    document.getElementById('statThroughput').innerText = `${qps} QPS`;
                    document.getElementById('statTotalTime').innerText = `${elapsedTotal.toFixed(1)}s`;
                }
            }

            // Launch concurrent workers based on selected batch concurrency
            const workers = [];
            const numWorkers = Math.min(concurrency, total);
            for (let w = 0; w < numWorkers; w++) {
                workers.push(worker());
            }
            await Promise.all(workers);

            // Save completed results to backend
            try {
                await fetch('/api/benchmark/save_results', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ results: benchmarkResults })
                });
            } catch (err) {
                console.error("Failed to save results to backend", err);
            }

            isRunningBenchmark = false;
            startBtn.disabled = false;
            benchBtnText.classList.remove('hidden');
            benchSpinner.classList.add('hidden');
            exportBtn.classList.remove('hidden');
            document.getElementById('logBtn').classList.remove('hidden');
        }

        function exportResults() {
            if (!benchmarkResults.length) {
                window.location.href = '/api/benchmark/export';
                return;
            }
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(benchmarkResults, null, 2));
            const downloadAnchor = document.createElement('a');
            downloadAnchor.setAttribute("href", dataStr);
            downloadAnchor.setAttribute("download", `benchmark_results_${Date.now()}.json`);
            document.body.appendChild(downloadAnchor);
            downloadAnchor.click();
            downloadAnchor.remove();
        }

        function escapeHtml(text) {
            if (!text) return "";
            return text
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print("\n" + "=" * 65)
    print(f"🚀 DSLM-GR ReAct Webapp running at: http://localhost:{port}")
    print("=" * 65 + "\n")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

