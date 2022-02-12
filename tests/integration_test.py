#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform integration tests for `orion.algo.skopt`."""
import itertools
import os

import numpy
import orion.core.cli
import pytest
from orion.core.utils import backward, format_trials
from orion.testing.algo import BaseAlgoTests

N_INIT = 5


class TestBOSkopt(BaseAlgoTests):
    max_trials = 100

    algo_name = "bayesianoptimizer"
    config = {
        "n_initial_points": N_INIT,
        "normalize_y": True,
        "acq_func": "EI",
        "alpha": 1e-8,
        "n_restarts_optimizer": 0,
        "noise": False,
        "seed": 1234,  # Because this is so random
        "convergence_duplicates": 5,
        "parallel_strategy": {
            "of_type": "StatusBasedParallelStrategy",
            "strategy_configs": {
                "broken": {"of_type": "MaxParallelStrategy", "default_result": 100},
            },
            "default_strategy": {
                "of_type": "meanparallelstrategy",
                "default_result": 50,
            },
        },
    }

    def test_is_done_cardinality(self):
        # TODO: Support correctly loguniform(discrete=True)
        #       See https://github.com/Epistimio/orion/issues/566
        space = self.update_space(
            {
                "x": "uniform(0, 4, discrete=True)",
                "y": "choices(['a', 'b', 'c'])",
                "z": "uniform(1, 6, discrete=True)",
            }
        )
        space = self.create_space(space)
        assert space.cardinality == 5 * 3 * 6

        algo = self.create_algo(space=space)
        for i, (x, y, z) in enumerate(itertools.product(range(5), "abc", range(1, 7))):
            assert not algo.is_done
            n = algo.n_suggested
            backward.algo_observe(
                algo,
                [format_trials.tuple_to_trial([x, y, z], space)],
                [dict(objective=i)],
            )
            assert algo.n_suggested == n + 1

        assert i + 1 == space.cardinality

        assert algo.is_done


TestBOSkopt.set_phases(
    [("random", 0, "space.sample"), ("bo", N_INIT + 1, "space.sample")]
)
