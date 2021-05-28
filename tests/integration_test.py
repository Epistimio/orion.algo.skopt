#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform integration tests for `orion.algo.skopt`."""
import itertools
import os

import numpy
import orion.core.cli
import pytest
from orion.testing.algo import BaseAlgoTests

N_INIT = 10


class TestBOSkopt(BaseAlgoTests):

    algo_name = "bayesianoptimizer"
    config = {
        "n_initial_points": N_INIT,
        "normalize_y": True,
        "acq_func": "EI",
        "alpha": 1e-8,
        "n_restarts_optimizer": 0,
        "noise": False,
        "strategy": None,
        "seed": 1234,  # Because this is so random
    }

    def test_suggest_init(self, mocker):
        algo = self.create_algo()
        spy = self.spy_phase(mocker, 0, algo, "space.sample")
        points = algo.suggest(1000)
        assert len(points) == N_INIT

    def test_suggest_init_missing(self, mocker):
        algo = self.create_algo()
        missing = 3
        spy = self.spy_phase(mocker, N_INIT - missing, algo, "space.sample")
        points = algo.suggest(1000)
        assert len(points) == missing

    def test_suggest_init_overflow(self, mocker):
        algo = self.create_algo()
        spy = self.spy_phase(mocker, N_INIT - 1, algo, "space.sample")
        # Now reaching N_INIT
        points = algo.suggest(1000)
        assert len(points) == 1
        # Verify point was sampled randomly, not using BO
        assert spy.call_count == 1
        # Overflow above N_INIT
        points = algo.suggest(1000)
        assert len(points) == 1
        # Verify point was sampled randomly, not using BO
        assert spy.call_count == 2

    def test_suggest_n(self, mocker, num, attr):
        algo = self.create_algo()
        spy = self.spy_phase(mocker, num, algo, attr)
        points = algo.suggest(5)
        if num == 0:
            assert len(points) == 5
        else:
            assert len(points) == 1

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
            n = len(algo.algorithm._trials_info)
            algo.observe([[x, y, z]], [dict(objective=i)])
            assert len(algo.algorithm._trials_info) == n + 1

        assert i + 1 == space.cardinality

        assert algo.is_done


TestBOSkopt.set_phases(
    [("random", 0, "space.sample"), ("bo", N_INIT + 1, "optimizer._ask")]
)
