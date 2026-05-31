"""
Diderot Effect Analysis — Pipeline Orchestrator

Runs all stages in order. Use --stage to run a single stage,
or omit it to run the full pipeline.

Usage:
    python pipeline.py                    # full pipeline
    python pipeline.py --stage embed      # embeddings only
    python pipeline.py --stage analyze    # content analysis + correlation + viz
    python pipeline.py --from-stage 4     # resume from stage 4 onward
"""

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STAGES = [
    ("load",        "initial_data_analysis",    "Load & filter raw CSV"),
    ("sort",        "diderot_sequential_sorting","Category mapping & sorting"),
    ("embed",       "embeddings",               "Aesthetic score embeddings"),
    ("triggers",    "identify_triggers",        "Phase labeling"),
    ("velocity",    "calculate_velocity",       "Inter-purchase intervals"),
    ("content",     "content_analysis",         "Keyword content analysis"),
    ("changepoint", "changepoint_analysis",     "Bayesian change point detection"),
    ("survival",    "survival_analysis",        "Cox proportional hazards model"),
    ("visualize",   "visualizations",            "Thesis visualizations"),
]

STAGE_NAMES = [s[0] for s in STAGES]


def run_stage(module_name: str, description: str):
    logger.info(f"{'='*60}")
    logger.info(f"Running: {description}")
    logger.info(f"{'='*60}")
    try:
        mod = __import__(module_name)
        mod.main()
    except Exception as e:
        logger.error(f"Stage '{module_name}' failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Diderot Effect Analysis Pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--stage",
        choices=STAGE_NAMES,
        help="Run a single named stage.",
    )
    group.add_argument(
        "--from-stage",
        type=int,
        metavar="N",
        help="Resume pipeline from stage N (1-indexed).",
    )
    args = parser.parse_args()

    if args.stage:
        idx = STAGE_NAMES.index(args.stage)
        stages_to_run = [STAGES[idx]]
    elif args.from_stage:
        n = args.from_stage - 1
        if not (0 <= n < len(STAGES)):
            logger.error(f"--from-stage must be between 1 and {len(STAGES)}.")
            sys.exit(1)
        stages_to_run = STAGES[n:]
    else:
        stages_to_run = STAGES

    logger.info(f"Running {len(stages_to_run)} stage(s)...")
    for name, module, description in stages_to_run:
        run_stage(module, description)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
