import json
import pickle
from pathlib import Path
from typing import List, Set, Dict, Any, Callable, Optional
import torch

from genret import config

class IDTrieNode:
    def __init__(self):
        self.children: Dict[int, IDTrieNode] = {}
        self.is_leaf: bool = False
        self.doc_ids: List[str] = []

class IDTrie:
    """
    Prefix Trie over all valid Semantic ID digit paths.
    Enables constrained beam search decoding during inference so the SLM can NEVER emit non-existent IDs.
    Also tracks subtree document IDs for Planning-Ahead (PAG) simultaneous scoring.
    """
    def __init__(self, id_items: Any):
        self.root = IDTrieNode()
        self._count = 0
        if isinstance(id_items, dict):
            for chunk_id, path in id_items.items():
                self.insert(path, chunk_id=chunk_id)
        else:
            for item in id_items:
                if isinstance(item, tuple) or isinstance(item, list) and len(item) == 2 and isinstance(item[0], list):
                    self.insert(item[0], chunk_id=item[1])
                else:
                    self.insert(item)

    def insert(self, path: List[int], chunk_id: Optional[str] = None):
        node = self.root
        if chunk_id is not None:
            node.doc_ids.append(chunk_id)
        for digit in path:
            if digit not in node.children:
                node.children[digit] = IDTrieNode()
            node = node.children[digit]
            if chunk_id is not None:
                node.doc_ids.append(chunk_id)
        node.is_leaf = True
        self._count += 1

    def allowed_next(self, prefix: List[int]) -> Set[int]:
        """
        Return the set of valid next digits for a given prefix path.
        Includes sentinel -1 if this prefix represents a complete valid ID.
        """
        node = self.root
        for digit in prefix:
            if digit not in node.children:
                return set()  # invalid prefix
            node = node.children[digit]

        allowed = set(node.children.keys())
        if node.is_leaf:
            allowed.add(-1)  # sentinel indicating end token </id> is allowed

        return allowed

    def get_subtree_docs(self, prefix: List[int]) -> List[str]:
        """
        Return all document chunk IDs situated within the subtree under prefix.
        Used by PAG (Planning-Ahead) to find max lookahead document prior scores.
        """
        node = self.root
        for digit in prefix:
            if digit not in node.children:
                return []
            node = node.children[digit]
        return node.doc_ids

    def is_complete(self, prefix: List[int]) -> bool:
        """Check if the given prefix corresponds to a complete valid chunk ID."""
        node = self.root
        for digit in prefix:
            if digit not in node.children:
                return False
            node = node.children[digit]
        return node.is_leaf

    def __len__(self) -> int:
        return self._count

    def save(self, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, in_path: Path) -> "IDTrie":
        with open(in_path, "rb") as f:
            return pickle.load(f)

def load_trie(ids_path: Path = config.IDS_PATH) -> IDTrie:
    """Build IDTrie with subtree doc mapping from data/ids.json."""
    if not ids_path.exists():
        raise FileNotFoundError(f"IDs file not found at {ids_path}. Run Section 4 (`python -m genret.semantic_ids`).")
    with open(ids_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return IDTrie(data["chunk_to_id"])

def make_prefix_allowed_tokens_fn(
    trie: IDTrie,
    tokenizer,
    digit_token_ids: Dict[int, int],
    id_start_token_id: int,
    id_end_token_id: int
) -> Callable[[int, torch.Tensor], List[int]]:
    """
    Constructs the `prefix_allowed_tokens_fn` callable required by HuggingFace `model.generate()`.
    
    Steps during generation:
    1. Scan `input_ids` from the end to find the most recent `id_start_token_id`.
    2. If `id_start_token_id` is not yet generated, allow `id_start_token_id`.
    3. Extract generated digit tokens, map to integer digits, and query `trie.allowed_next(prefix)`.
    4. Map allowed digits back to vocabulary token IDs.
    5. If -1 is allowed (sentinel for completed ID), add `id_end_token_id`.
    """
    token_to_digit = {tok_id: digit for digit, tok_id in digit_token_ids.items()}

    def prefix_allowed_tokens(batch_id: int, input_ids: torch.Tensor) -> List[int]:
        ids_list = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else list(input_ids)

        # Find the position of the opening <id> token
        if id_start_token_id not in ids_list:
            return [id_start_token_id]

        start_idx = len(ids_list) - 1 - ids_list[::-1].index(id_start_token_id)
        generated_id_tokens = ids_list[start_idx + 1:]

        # If </id> already emitted, only allow EOS / pad
        if id_end_token_id in generated_id_tokens:
            eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else id_end_token_id
            return [eos_id]

        # Extract digit sequence generated so far
        current_digits = []
        for tok in generated_id_tokens:
            if tok in token_to_digit:
                current_digits.append(token_to_digit[tok])
            else:
                return [id_end_token_id]  # fallback on unknown

        allowed_digits = trie.allowed_next(current_digits)
        allowed_token_ids = []

        for d in allowed_digits:
            if d == -1:
                allowed_token_ids.append(id_end_token_id)
            elif d in digit_token_ids:
                allowed_token_ids.append(digit_token_ids[d])

        # Fallback if no allowed tokens (force end)
        if not allowed_token_ids:
            return [id_end_token_id]

        return allowed_token_ids

    return prefix_allowed_tokens

def run_acceptance_test(trie: IDTrie, ids_data: Dict[str, Any]):
    """
    Section 5 Acceptance Test:
    - len(trie) == number of chunks
    - For every valid ID, walking digit by digit is always allowed and END is allowed at final position
    - Invalid digit at any position is disallowed
    """
    print("\n" + "=" * 60)
    print("SECTION 5 ACCEPTANCE TEST: PREFIX TRIE CONSTRAINTS")
    print("=" * 60)

    id_lists = list(ids_data["chunk_to_id"].values())

    # 1. Assert size
    assert len(trie) == len(id_lists), f"Trie size mismatch: {len(trie)} != {len(id_lists)}"
    print(f"✓ Trie Size Passed: {len(trie)} distinct paths registered.")

    # 2. Assert path traversal and completion
    for path in id_lists:
        prefix = []
        for i, digit in enumerate(path):
            allowed = trie.allowed_next(prefix)
            assert digit in allowed, f"Digit {digit} not allowed at prefix {prefix}"
            prefix.append(digit)
        
        # At the end of the path, sentinel -1 (END) must be allowed
        final_allowed = trie.allowed_next(prefix)
        assert -1 in final_allowed, f"END sentinel not allowed at completed path {prefix}"
        assert trie.is_complete(prefix), f"trie.is_complete({prefix}) returned False for valid ID"

    print("✓ All Valid ID Paths Successfully Traversed & Verified.")

    # 3. Assert invalid paths disallowed
    invalid_prefix = [999, 999]
    assert trie.allowed_next(invalid_prefix) == set(), "Invalid prefix returned allowed tokens!"
    print("✓ Disallowed Path Constraint Passed: Non-existent prefix returns empty set.")

    print("\n✅ Section 5 Acceptance Test Passed!")

if __name__ == "__main__":
    with open(config.IDS_PATH, "r") as f:
        ids_data = json.load(f)
    trie = IDTrie(list(ids_data["chunk_to_id"].values()))
    run_acceptance_test(trie, ids_data)
