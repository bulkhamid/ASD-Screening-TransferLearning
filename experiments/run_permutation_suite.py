#!/usr/bin/env python
"""Run the planned grouped training-label permutation controls."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from experiments.run_subject_benchmark import run


def main():
    parser = argparse.ArgumentParser(
        description="Five Simple2DCNN permutations and one complete permutation per larger model."
    )
    parser.add_argument("--cache_dir", default="data_grouped")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--config", default="configs/subject_cv.json")
    parser.add_argument("--output_dir", default="runs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--wandb_mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--wandb_project", default="ASD-SubjectCV-Permutations")
    args = parser.parse_args()

    plans = [
        *(('simple2dcnn', seed) for seed in (1042, 1043, 1044, 1045, 1046)),
        ("simple3dlstm_v2", 2042),
        ("r3d", 2042),
    ]
    completed = []
    for model, permutation_seed in plans:
        run_args = SimpleNamespace(
            cache_dir=args.cache_dir,
            manifest=args.manifest,
            config=args.config,
            output_dir=args.output_dir,
            models=[model],
            seeds=args.seeds,
            folds=args.folds,
            device=args.device,
            no_pretrained=False,
            permutation_seed=permutation_seed,
            temporal_ablations=["normal"],
            wandb_mode=args.wandb_mode,
            wandb_project=args.wandb_project,
            validate_only=False,
            epoch_limit=None,
        )
        completed.append(
            {
                "model": model,
                "permutation_seed": permutation_seed,
                "run_dir": str(run(run_args)),
            }
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output_dir) / f"permutation_suite_{timestamp}.json"
    output.write_text(json.dumps(completed, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
