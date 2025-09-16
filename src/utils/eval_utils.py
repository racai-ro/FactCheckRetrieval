from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)
from sentence_transformers.util import cos_sim


def create_evaluator(args, test_queries, data, test_pairs):
    """Create the evaluator based on method and matryoshka setting."""
    if args.method == 'lora' and args.use_matryoshka:
        print('Creating Matryoshka evaluators')
        matryoshka_dimensions = [1024, 768, 512, 256, 128, 64]
        matryoshka_evaluators = []

        for dim in matryoshka_dimensions:
            ir_evaluator = InformationRetrievalEvaluator(
                queries=test_queries,
                corpus=data['finetune_corpus'],
                relevant_docs=test_pairs,
                name=f"dim_{dim}",
                truncate_dim=dim,
                score_functions={"cosine": cos_sim},
                show_progress_bar=False
            )
            matryoshka_evaluators.append(ir_evaluator)

        evaluator = SequentialEvaluator(matryoshka_evaluators)
        metric_for_best_model = "eval_dim_1024_cosine_ndcg@10"
    else:
        print('Creating standard evaluator')
        evaluator = InformationRetrievalEvaluator(
            queries=test_queries,
            corpus=data['finetune_corpus'],
            relevant_docs=test_pairs,
            name="gemma",
            score_functions={"cosine": cos_sim},
            show_progress_bar=False,
        )
        metric_for_best_model = "eval_gemma_cosine_ndcg@10"
    return evaluator, metric_for_best_model
