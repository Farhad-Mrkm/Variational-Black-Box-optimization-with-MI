#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np

from vbo_mi import VBOConfig, VBOMI
from vbo_mi.objectives import paper_synthetic_objective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="ackley", choices=["ackley", "levy", "griewank", "hartmann6", "rastrigin"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=70)
    parser.add_argument("--out", type=Path, default=Path("results/multiseed"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        objective = paper_synthetic_objective(args.task)
        cfg = VBOConfig(seed=seed)
        result = VBOMI(objective, cfg).run(
            batch_size=args.batch_size,
            warmup_iterations=args.warmup,
            main_iterations=args.iterations,
        )
        result.to_csv(args.out / f"{args.task}_seed_{seed}.csv")

    # Aggregate only curves of equal length.
    runs = []
    for seed in args.seeds:
        arr = np.loadtxt(args.out / f"{args.task}_seed_{seed}.csv", delimiter=",", skiprows=1, usecols=2)
        runs.append(arr)
    arr = np.stack(runs)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    np.savetxt(args.out / f"{args.task}_mean_std.csv", np.c_[mean, std], delimiter=",", header="mean,std", comments="")
    print(f"Saved {len(args.seeds)} seeds for {args.task} in {args.out}")


if __name__ == "__main__":
    main()
