#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Benchmark all algorithms on the rosenbrock function."""
import argparse

import orion.core.cli
from orion.benchmark.benchmark_client import get_or_create_benchmark
from orion.benchmark.assessment import AverageResult
from orion.benchmark.task import RosenBrock, EggHolder, CarromTable, Branin


def main(argv=None):
    """Execute the benchmark with cli"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Number of repetitions for each benchmark.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=100,
        help="Number of trials for each run",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Folder where to save figures.",
    )

    options = parser.parse_args(argv)

    benchmark = get_or_create_benchmark(
        # TODO Add template here
        name="benchmark_bayesian_optimizer",
        # TODO Add template here
        algorithms=["random", "bayesianoptimizer"],
        debug=True,  # Use EphemeralDB
        targets=[
            {
                "assess": [
                    AverageResult(options.repetitions),
                ],
                "task": [
                    Branin(options.max_trials)n
                    # RosenBrock(options.max_trials, dim=3),
                    # EggHolder(options.max_trials, dim=4),
                    # CarromTable(options.max_trials),
                ],
            }
        ],
    )

    benchmark.process()
    for study, figure in zip(benchmark.studies, benchmark.analysis()):
        figure.write_html(
            f"{options.output}/{study.task_name}_{study.assess_name}.html"
        )


if __name__ == "__main__":
    main()
