"""Baseline 6: RAG-Based Biomedical Agent for continual KGQA.

Uses retrieval-augmented generation over an evolving biomedical knowledge graph.
Key insight: By updating the retrieval index (not model weights), RAG naturally
avoids catastrophic forgetting. New knowledge is added to the vector store
without modifying the LLM. However, may suffer from retrieval drift.

Supports two modes:
1. Full RAG: ChromaDB retrieval + LLM generation (requires GPU for LLM)
2. Retrieval-only: ChromaDB retrieval + direct entity extraction (no LLM needed)

Usage:
    from src.baselines.rag_agent import BiomedicalRAGAgent
    agent = BiomedicalRAGAgent()
    agent.index_kg_snapshot(triples)
    answer = agent.answer_question("What drugs treat disease X?")
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _triple_to_sentence(head: str, relation: str, tail: str) -> str:
    """Convert a KG triple to a natural language sentence for indexing.

    Args:
        head: Head entity ID.
        relation: Relation type.
        tail: Tail entity ID.

    Returns:
        Natural language sentence.
    """
    # Clean entity names
    h = head.replace("_", " ")
    t = tail.replace("_", " ")
    r = relation.replace("_", " ")
    return f"{h} has {r} relationship with {t}."


class BiomedicalRAGAgent:
    """RAG-based agent for continual biomedical KGQA.

    Uses ChromaDB as the vector store with a biomedical sentence
    transformer for embeddings. Optionally uses an LLM (e.g., Llama-3-8B)
    for answer generation, with fallback to retrieval-only mode.

    Args:
        llm_name: HuggingFace model ID for text generation.
        embedding_model: Sentence embedding model for ChromaDB.
        persist_dir: Directory for ChromaDB persistence.
        use_llm: Whether to load and use the LLM. If False, uses
            retrieval-only mode (extracts answers from retrieved context).
    """

    def __init__(
        self,
        llm_name: str = "Qwen/Qwen2.5-7B-Instruct",
        embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        persist_dir: Optional[str] = None,
        use_llm: bool = True,
        no_retrieval: bool = False,
    ) -> None:
        self.llm_name = llm_name
        self.embedding_model_name = embedding_model
        self.persist_dir = persist_dir
        self.use_llm = use_llm
        self.no_retrieval = no_retrieval
        self.vectorstore = None
        self.llm = None
        self._embedding_fn = None
        self._collection = None
        self._doc_count = 0

    def _init_vectorstore(self) -> None:
        """Initialize ChromaDB client and collection with embedding function."""
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is required for RAG agent. "
                "Install with: pip install chromadb"
            )

        # Use SentenceTransformer embedding function
        try:
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )
            self._embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name,
            )
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}. "
                           "Using ChromaDB default embeddings.")
            self._embedding_fn = None

        if self.persist_dir:
            client = chromadb.PersistentClient(path=self.persist_dir)
        else:
            client = chromadb.Client()

        # Create or get collection
        kwargs = {"name": "kg_triples"}
        if self._embedding_fn is not None:
            kwargs["embedding_function"] = self._embedding_fn

        self._collection = client.get_or_create_collection(**kwargs)
        self._doc_count = self._collection.count()
        logger.info(f"ChromaDB initialized (existing docs: {self._doc_count})")

    def _init_llm(self) -> None:
        """Initialize HuggingFace text-generation pipeline.

        Falls back to retrieval-only mode if LLM loading fails.
        """
        if not self.use_llm:
            logger.info("LLM disabled, using retrieval-only mode")
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            logger.info(f"Loading LLM: {self.llm_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                self.llm_name, trust_remote_code=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.llm_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
            self.llm = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=128,
                do_sample=False,
            )
            logger.info(f"LLM loaded: {self.llm_name}")
        except Exception as e:
            logger.warning(f"Failed to load LLM: {e}. "
                           "Falling back to retrieval-only mode.")
            self.llm = None
            self.use_llm = False

    def index_kg_snapshot(
        self,
        kg_triples: np.ndarray | list[tuple],
        batch_size: int = 5000,
        id_to_entity: dict[int, str] | None = None,
        id_to_relation: dict[int, str] | None = None,
    ) -> int:
        """Index a KG snapshot into the vector store.

        Each triple is converted to natural language and indexed.

        Args:
            kg_triples: Array/list of (head, relation, tail) triples.
                Can be int IDs (requires id_to_entity/id_to_relation) or strings.
            batch_size: Number of documents to add per batch.
            id_to_entity: Reverse mapping from int ID to entity name string.
            id_to_relation: Reverse mapping from int ID to relation name string.

        Returns:
            Number of documents indexed.
        """
        if self._collection is None:
            self._init_vectorstore()

        documents = []
        ids = []
        metadatas = []

        for i, triple in enumerate(kg_triples):
            h_raw, r_raw, t_raw = triple[0], triple[1], triple[2]
            # Convert int IDs to strings if mappings provided
            if id_to_entity is not None:
                h = id_to_entity.get(int(h_raw), str(h_raw))
                t = id_to_entity.get(int(t_raw), str(t_raw))
            else:
                h, t = str(h_raw), str(t_raw)
            if id_to_relation is not None:
                r = id_to_relation.get(int(r_raw), str(r_raw))
            else:
                r = str(r_raw)
            doc = _triple_to_sentence(h, r, t)
            doc_id = f"doc_{self._doc_count + i}"
            documents.append(doc)
            ids.append(doc_id)
            metadatas.append({"head": h, "relation": r, "tail": t})

        # Add in batches to avoid ChromaDB limits
        total_added = 0
        for start in range(0, len(documents), batch_size):
            end = min(start + batch_size, len(documents))
            self._collection.add(
                documents=documents[start:end],
                ids=ids[start:end],
                metadatas=metadatas[start:end],
            )
            total_added += end - start

        self._doc_count += total_added
        logger.info(f"Indexed {total_added} triples (total: {self._doc_count})")
        return total_added

    def update_with_new_knowledge(
        self,
        new_triples: np.ndarray | list[tuple],
        batch_size: int = 5000,
        id_to_entity: dict[int, str] | None = None,
        id_to_relation: dict[int, str] | None = None,
    ) -> int:
        """Continual update: add new knowledge to vector store.

        No retraining needed - just add new documents to the index.
        This is the key advantage over parametric CL methods.

        Args:
            new_triples: New triples to add.
            batch_size: Batch size for indexing.
            id_to_entity: Reverse mapping from int ID to entity name string.
            id_to_relation: Reverse mapping from int ID to relation name string.

        Returns:
            Number of new documents added.
        """
        return self.index_kg_snapshot(
            new_triples, batch_size=batch_size,
            id_to_entity=id_to_entity, id_to_relation=id_to_relation,
        )

    def answer_question(
        self,
        question: str,
        k: int = 10,
    ) -> str:
        """Answer a biomedical question using RAG over the KG.

        1. Retrieve relevant triples from vector store.
        2. If LLM available: feed context + question to LLM.
        3. If no LLM: extract most common tail entity from retrieved triples.

        Args:
            question: Natural language question.
            k: Number of documents to retrieve.

        Returns:
            Generated answer string.
        """
        if self._collection is None:
            self._init_vectorstore()

        # Retrieve relevant documents
        results = self._collection.query(
            query_texts=[question],
            n_results=min(k, self._doc_count) if self._doc_count > 0 else 1,
        )

        if not results["documents"] or not results["documents"][0]:
            return ""

        retrieved_docs = results["documents"][0]
        retrieved_meta = results["metadatas"][0] if results["metadatas"] else []

        if self.llm is not None:
            return self._generate_with_llm(question, retrieved_docs)
        else:
            return self._extract_from_retrieval(question, retrieved_meta)

    def answer_question_zero_shot(self, question: str) -> str:
        """Answer a question using only the LLM, without retrieval.

        Args:
            question: Natural language question.

        Returns:
            LLM-generated answer.
        """
        if self.llm is None:
            return ""
        prompt = (
            f"Answer this biomedical question concisely in one phrase:\n"
            f"{question}\n"
            f"Answer:"
        )
        output = self.llm(prompt, return_full_text=False)
        answer = output[0]["generated_text"].strip().split("\n")[0].strip()
        return answer

    def _generate_with_llm(
        self,
        question: str,
        context_docs: list[str],
    ) -> str:
        """Generate answer using the LLM with retrieved context.

        Args:
            question: The question to answer.
            context_docs: Retrieved context documents.

        Returns:
            LLM-generated answer.
        """
        context = "\n".join(context_docs[:10])
        prompt = (
            f"Based on the following biomedical knowledge:\n\n"
            f"{context}\n\n"
            f"Answer this question concisely: {question}\n"
            f"Answer:"
        )
        output = self.llm(prompt, return_full_text=False)
        answer = output[0]["generated_text"].strip()
        # Take only the first line/sentence
        answer = answer.split("\n")[0].strip()
        return answer

    def _extract_from_retrieval(
        self,
        question: str,
        metadatas: list[dict],
    ) -> str:
        """Extract answer directly from retrieved triple metadata.

        Uses majority voting: the most commonly appearing tail entity
        in retrieved triples is the answer.

        Args:
            question: The question (used for context).
            metadatas: Metadata dicts from retrieved documents.

        Returns:
            Most common tail entity from retrieved triples.
        """
        if not metadatas:
            return ""

        # Collect tail entities from retrieved triples
        tails = [m.get("tail", "") for m in metadatas if m.get("tail")]
        if not tails:
            return ""

        # Return most common tail, cleaned to match gold answer format
        from src.data.kgqa import _clean_entity_name

        counter = Counter(tails)
        answer = counter.most_common(1)[0][0]
        return _clean_entity_name(answer)

    def evaluate_kgqa(
        self,
        qa_pairs: list[dict[str, str]],
    ) -> dict[str, float]:
        """Batch evaluate the agent on a set of QA pairs.

        Computes Exact Match (EM) and token-level F1 for each question.

        Args:
            qa_pairs: List of dicts with 'question' and 'answer' keys.

        Returns:
            Dict with 'exact_match', 'token_f1', 'accuracy' metrics.
        """
        if not qa_pairs:
            return {"exact_match": 0.0, "token_f1": 0.0, "accuracy": 0.0}

        ems = []
        f1s = []

        for qa in qa_pairs:
            question = qa["question"]
            gold = qa["answer"]
            if self.no_retrieval:
                predicted = self.answer_question_zero_shot(question)
            else:
                predicted = self.answer_question(question)

            em = compute_exact_match(predicted, gold)
            f1 = compute_token_f1(predicted, gold)
            ems.append(em)
            f1s.append(f1)

        return {
            "exact_match": float(np.mean(ems)),
            "token_f1": float(np.mean(f1s)),
            "accuracy": float(np.mean(ems)),
        }


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match between prediction and ground truth.

    Normalizes both strings (lowercase, strip whitespace/punctuation)
    before comparison.

    Args:
        prediction: Predicted answer string.
        ground_truth: Gold answer string.

    Returns:
        1.0 if exact match, 0.0 otherwise.
    """
    pred_norm = _normalize_answer(prediction)
    gold_norm = _normalize_answer(ground_truth)
    return 1.0 if pred_norm == gold_norm else 0.0


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth.

    Tokenizes both strings and computes precision/recall/F1 based on
    token overlap.

    Args:
        prediction: Predicted answer string.
        ground_truth: Gold answer string.

    Returns:
        Token F1 score in [0, 1].
    """
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(ground_truth).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def _normalize_answer(s: str) -> str:
    """Normalize answer string for comparison.

    Lowercases, removes punctuation, strips whitespace, removes articles.

    Args:
        s: Raw answer string.

    Returns:
        Normalized string.
    """
    s = s.lower().strip()
    # Remove punctuation
    s = re.sub(r"[^\w\s]", "", s)
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Collapse whitespace
    s = " ".join(s.split())
    return s
