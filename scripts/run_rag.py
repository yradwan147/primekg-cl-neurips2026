"""Run RAG agent baseline for continual KGQA evaluation.

Indexes each KG task snapshot into ChromaDB, generates KGQA questions
from test triples, and evaluates the agent on all tasks seen so far.

Usage:
    # Quick mode: 10 questions, retrieval-only (no LLM)
    python scripts/run_rag.py --quick

    # Full run with LLM
    python scripts/run_rag.py --seeds 42 123 456 789 1024

    # Retrieval-only (no LLM needed)
    python scripts/run_rag.py --no-llm --questions-per-task 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 1024]


def run_rag_evaluation(
    task_seq: dict,
    task_names: list[str],
    questions_per_task: int,
    use_llm: bool,
    llm_name: str,
    embedding_model: str,
    seed: int,
    entity_to_id: dict[str, int] | None = None,
    relation_to_id: dict[str, int] | None = None,
    no_retrieval: bool = False,
) -> dict:
    """Run RAG evaluation across all tasks for one seed.

    Args:
        task_seq: Task sequence data.
        task_names: Ordered task names.
        questions_per_task: Number of QA pairs per task.
        use_llm: Whether to use LLM for generation.
        llm_name: HuggingFace model ID.
        embedding_model: Sentence embedding model name.
        seed: Random seed.

    Returns:
        Dict with results_matrix, per-task metrics, CL metrics.
    """
    from src.baselines.rag_agent import BiomedicalRAGAgent
    from src.data.kgqa import generate_kgqa_questions
    from src.evaluation.metrics import evaluate_continual_learning

    np.random.seed(seed)

    # Build reverse mappings for int ID -> string conversion
    id_to_entity = {v: k for k, v in entity_to_id.items()} if entity_to_id else None
    id_to_relation = {v: k for k, v in relation_to_id.items()} if relation_to_id else None

    # Initialize agent (new for each seed for clean evaluation)
    agent = BiomedicalRAGAgent(
        llm_name=llm_name,
        embedding_model=embedding_model,
        use_llm=use_llm,
        no_retrieval=no_retrieval,
    )

    num_tasks = len(task_names)
    # R[i][j] = token_f1 on task j after indexing through task i
    results_matrix = np.zeros((num_tasks, num_tasks))

    # Generate QA questions for each task from test triples
    qa_per_task = {}
    for task_name, task_data in task_seq.items():
        qa_per_task[task_name] = generate_kgqa_questions(
            task_data["test"], n=questions_per_task, seed=seed,
            id_to_entity=id_to_entity, id_to_relation=id_to_relation,
        )

    for task_idx, task_name in enumerate(task_names):
        logger.info(f"=== Task {task_idx + 1}/{num_tasks}: {task_name} ===")
        task_data = task_seq[task_name]

        # Index training triples for this task
        if task_idx == 0:
            agent.index_kg_snapshot(
                task_data["train"],
                id_to_entity=id_to_entity, id_to_relation=id_to_relation,
            )
        else:
            agent.update_with_new_knowledge(
                task_data["train"],
                id_to_entity=id_to_entity, id_to_relation=id_to_relation,
            )

        # Evaluate on all tasks seen so far
        for eval_idx in range(task_idx + 1):
            eval_name = task_names[eval_idx]
            qa_pairs = qa_per_task[eval_name]

            metrics = agent.evaluate_kgqa(qa_pairs)
            # Use token_f1 as the primary metric for the results matrix
            results_matrix[task_idx, eval_idx] = metrics["token_f1"]
            logger.info(f"  Eval {eval_name}: F1={metrics['token_f1']:.4f}, "
                        f"EM={metrics['exact_match']:.4f}")

    # Compute CL metrics
    cl_metrics = evaluate_continual_learning(results_matrix, task_names)

    return {
        "results_matrix": results_matrix.tolist(),
        "task_names": task_names,
        "cl_metrics": cl_metrics,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG agent for KGQA")
    parser.add_argument("--tasks-dir", default="data/benchmark/tasks")
    parser.add_argument("--task-names", nargs="+", default=None)
    parser.add_argument("--llm", default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace LLM model ID")
    parser.add_argument("--embedding-model",
                        default="pritamdeka/S-PubMedBert-MS-MARCO")
    parser.add_argument("--questions-per-task", type=int, default=200)
    parser.add_argument("--no-llm", action="store_true",
                        help="Use retrieval-only mode (no LLM)")
    parser.add_argument("--no-retrieval", action="store_true",
                        help="Zero-shot LLM mode (no retrieval, LLM answers directly)")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--output-suffix", default="",
                        help="Suffix for output filename (e.g. _seed42)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--eval-multihop", action="store_true",
                        help="Run multi-hop RAG evaluation after standard eval")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 10 questions, retrieval-only")
    args = parser.parse_args()

    if args.quick:
        args.questions_per_task = 10
        args.no_llm = True
        logger.info("Quick mode: 10 questions, retrieval-only")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.baselines._base import load_task_sequence

    # Load tasks
    task_seq, entity_to_id, relation_to_id = load_task_sequence(
        args.tasks_dir, args.task_names
    )
    task_names = list(task_seq.keys())

    logger.info(f"Tasks: {task_names}")
    logger.info(f"Entities: {len(entity_to_id):,}, Relations: {len(relation_to_id)}")
    logger.info(f"Questions per task: {args.questions_per_task}")
    logger.info(f"LLM: {'disabled' if args.no_llm else args.llm}")

    all_seed_results = []
    print(f"[STARTED] method=rag mode={'retrieval_only' if args.no_llm else 'full_rag'} "
          f"seeds={args.seeds} tasks={len(task_names)} questions={args.questions_per_task}")

    for seed in args.seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"RAG Seed {seed}")
        logger.info(f"{'='*60}")
        start = time.time()

        result = run_rag_evaluation(
            task_seq=task_seq,
            task_names=task_names,
            questions_per_task=args.questions_per_task,
            use_llm=not args.no_llm,
            llm_name=args.llm,
            embedding_model=args.embedding_model,
            seed=seed,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            no_retrieval=args.no_retrieval,
        )

        elapsed = time.time() - start
        result["training_time_s"] = elapsed

        cl = result["cl_metrics"]
        logger.info(f"  AP={cl['Average Performance (AP)']:.4f}, "
                    f"AF={cl['Average Forgetting (AF)']:.4f}, "
                    f"REM={cl['Remembering (REM)']:.4f} ({elapsed:.0f}s)")

        all_seed_results.append(result)
        print(f"[PROGRESS] method=rag seed={seed} "
              f"AP={cl['Average Performance (AP)']:.4f} "
              f"AF={cl['Average Forgetting (AF)']:.4f} "
              f"elapsed={elapsed:.0f}s")

        # Save after each seed so partial results survive failures
        mode = "retrieval_only" if args.no_llm else "full_rag"
        result_path = output_dir / f"rag_{mode}{args.output_suffix}.json"
        with open(result_path, "w") as f:
            json.dump({
                "method": "rag_agent",
                "mode": mode,
                "llm": args.llm if not args.no_llm else None,
                "embedding_model": args.embedding_model,
                "questions_per_task": args.questions_per_task,
                "task_names": task_names,
                "seeds": args.seeds,
                "results": all_seed_results,
            }, f, indent=2, default=str)
        logger.info(f"Intermediate save: {result_path} ({len(all_seed_results)}/{len(args.seeds)} seeds)")

    # Multi-hop RAG evaluation (if requested)
    multihop_results = None
    if args.eval_multihop:
        from src.evaluation.multihop import (
            extract_all_path_types,
            evaluate_multihop_rag,
        )

        logger.info("Running multi-hop RAG evaluation...")
        all_train = np.concatenate(
            [data["train"] for data in task_seq.values()], axis=0,
        )
        all_paths = extract_all_path_types(
            all_train, relation_to_id, max_paths_per_type=2000,
        )
        id_to_entity = {v: k for k, v in entity_to_id.items()}
        id_to_relation = {v: k for k, v in relation_to_id.items()}

        # Use the last seed's agent (still in memory) for multi-hop eval
        from src.baselines.rag_agent import BiomedicalRAGAgent
        mh_agent = BiomedicalRAGAgent(
            llm_name=args.llm,
            embedding_model=args.embedding_model,
            use_llm=not args.no_llm,
        )
        # Index all training data
        mh_agent.index_kg_snapshot(
            all_train,
            id_to_entity=id_to_entity, id_to_relation=id_to_relation,
        )

        multihop_results = {}
        for desc, paths in all_paths.items():
            if not paths:
                continue
            mh_metrics = evaluate_multihop_rag(
                mh_agent, paths, id_to_entity, id_to_relation,
                max_questions=200,
            )
            multihop_results[desc] = mh_metrics
            logger.info(f"  {desc}: EM={mh_metrics['multihop_EM']:.4f}, "
                        f"F1={mh_metrics['multihop_F1']:.4f} "
                        f"({mh_metrics['num_questions']} questions)")

    # Save results
    mode = "retrieval_only" if args.no_llm else "full_rag"
    result_path = output_dir / f"rag_{mode}{args.output_suffix}.json"
    output_data = {
        "method": "rag_agent",
        "mode": mode,
        "llm": args.llm if not args.no_llm else None,
        "embedding_model": args.embedding_model,
        "questions_per_task": args.questions_per_task,
        "task_names": task_names,
        "seeds": args.seeds,
        "results": all_seed_results,
    }
    if multihop_results:
        output_data["multihop_results"] = multihop_results
    with open(result_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    logger.info(f"Results saved to {result_path}")


if __name__ == "__main__":
    try:
        main()
        print("[SUCCESS] run_rag completed")
    except Exception as e:
        print(f"[FAILED] run_rag error={str(e)[:200]}")
        raise
