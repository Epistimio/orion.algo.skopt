#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform integration tests for `orion.algo.skopt`."""
import os

import numpy
import orion.core.cli
import pytest
from orion.algo.space import Integer, Real, Space
from orion.client import create_experiment
from orion.core.worker.primary_algo import PrimaryAlgo
from orion.testing import OrionState


@pytest.fixture()
def space():
    """Return an optimization space"""
    space = Space()
    dim1 = Integer("yolo1", "uniform", -3, 6)
    space.register(dim1)
    dim2 = Real("yolo2", "uniform", 0, 1)
    space.register(dim2)

    return space


@pytest.fixture()
def real_space():
    """Return a real optimization space"""
    space = Space()
    dim1 = Real("yolo1", "uniform", -3, 6)
    space.register(dim1)
    dim2 = Real("yolo2", "uniform", 0, 1)
    space.register(dim2)

    return space


def test_seeding(space):
    """Verify that seeding makes sampling deterministic"""
    bayesian_optimizer = PrimaryAlgo(
        space, {"bayesianoptimizer": {"n_initial_points": 3}}
    )

    bayesian_optimizer.seed_rng(1)
    a = bayesian_optimizer.suggest(1)[0]
    with pytest.raises(AssertionError):
        numpy.testing.assert_equal(a, bayesian_optimizer.suggest(1)[0])

    bayesian_optimizer.seed_rng(1)
    numpy.testing.assert_equal(a, bayesian_optimizer.suggest(1)[0])


def test_seeding_bo(space):
    """Verify that seeding makes bayesian optimization deterministic"""
    bayesian_optimizer = PrimaryAlgo(
        space, {"bayesianoptimizer": {"n_initial_points": 3}}
    )

    bayesian_optimizer.seed_rng(1)
    for i in range(5):
        a = bayesian_optimizer.suggest(1)[0]
        bayesian_optimizer.observe([a], [{"objective": i}])
    with pytest.raises(AssertionError):
        numpy.testing.assert_equal(a, bayesian_optimizer.suggest(1)[0])

    # Same seed, should be equal
    bayesian_optimizer = PrimaryAlgo(
        space, {"bayesianoptimizer": {"n_initial_points": 3}}
    )
    bayesian_optimizer.seed_rng(1)
    for i in range(5):
        b = bayesian_optimizer.suggest(1)[0]
        bayesian_optimizer.observe([b], [{"objective": i}])
    numpy.testing.assert_equal(a, b)

    # Not same seed, should diverge
    bayesian_optimizer = PrimaryAlgo(
        space, {"bayesianoptimizer": {"n_initial_points": 3}}
    )
    bayesian_optimizer.seed_rng(2)
    for i in range(5):
        c = bayesian_optimizer.suggest(1)[0]
        bayesian_optimizer.observe([c], [{"objective": i}])
    with pytest.raises(AssertionError):
        numpy.testing.assert_equal(a, c)


def test_set_state(space):
    """Verify that resetting state makes sampling deterministic"""
    bayesian_optimizer = PrimaryAlgo(space, "bayesianoptimizer")

    state = bayesian_optimizer.state_dict
    a = bayesian_optimizer.suggest(1)[0]
    with pytest.raises(AssertionError):
        numpy.testing.assert_equal(a, bayesian_optimizer.suggest(1)[0])

    bayesian_optimizer.set_state(state)
    numpy.testing.assert_equal(a, bayesian_optimizer.suggest(1)[0])


def test_set_state_bo(real_space):
    """Verify that resetting state during BO makes sampling deterministic"""
    n_init = 3
    optimizer = PrimaryAlgo(
        real_space,
        {
            "bayesianoptimizer": {
                "n_initial_points": 3,
                "normalize_y": True,
                "acq_func": "EI",
                "noise": None,
            }
        },
    )

    for i in range(n_init + 2):
        a = optimizer.suggest(1)[0]
        if i < n_init:
            assert getattr(optimizer.algorithm.optimizer, "_next_x", None) is None
        optimizer.observe([a], [{"objective": i / (n_init + 2)}])

    # Make sure we left random regime and are now doing sampling
    assert optimizer.algorithm.optimizer._next_x is not None
    assert optimizer.algorithm.optimizer._n_initial_points <= 0

    # Make sure BO returns different samples during iterations
    state = optimizer.state_dict
    a = optimizer.suggest(1)[0]
    optimizer.observe([a], [{"objective": i + 1 / (n_init + 3)}])

    with pytest.raises(AssertionError):
        numpy.testing.assert_equal(a, optimizer.suggest(1)[0])

    # Reset state and make sure BO returns the same sample
    optimizer.set_state(state)

    assert optimizer.algorithm.optimizer._next_x is not None
    assert optimizer.algorithm.optimizer._n_initial_points <= 0

    numpy.testing.assert_equal(a, optimizer.suggest(1)[0])

    # Test also when setting state to new optimizer, not old version.
    new_optimizer = PrimaryAlgo(
        real_space,
        {
            "bayesianoptimizer": {
                "n_initial_points": 3,
                "normalize_y": True,
                "acq_func": "EI",
                "noise": None,
            }
        },
    )

    new_optimizer.set_state(state)

    assert new_optimizer.algorithm.optimizer._next_x is not None
    assert new_optimizer.algorithm.optimizer._n_initial_points <= 0

    numpy.testing.assert_equal(a, new_optimizer.suggest(1)[0])


def test_bayesian_optimizer_basic(monkeypatch):
    """Check functionality of BayesianOptimizer wrapper for single shaped dimension."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    with OrionState(experiments=[], trials=[]):

        orion.core.cli.main(
            [
                "hunt",
                "--config",
                "./orion_config_bayes.yaml",
                "./rosenbrock.py",
                "-x~uniform(-5, 5)",
            ]
        )


def test_int(monkeypatch):
    """Check support of integer values."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    with OrionState(experiments=[], trials=[]):

        orion.core.cli.main(
            [
                "hunt",
                "--name",
                "exp",
                "--max-trials",
                "5",
                "--config",
                "./orion_config_bayes.yaml",
                "./rosenbrock.py",
                "-x~uniform(-5, 5, discrete=True)",
            ]
        )


def test_categorical(monkeypatch):
    """Check support of categorical values."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    with OrionState(experiments=[], trials=[]):

        orion.core.cli.main(
            [
                "hunt",
                "--name",
                "exp",
                "--max-trials",
                "5",
                "--config",
                "./orion_config_bayes.yaml",
                "./rosenbrock.py",
                "-x~choices([-5, -2, 0, 2, 5])",
            ]
        )


def test_linear(monkeypatch):
    """Check support of logarithmic distributions."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    with OrionState(experiments=[], trials=[]):

        orion.core.cli.main(
            [
                "hunt",
                "--name",
                "exp",
                "--max-trials",
                "5",
                "--config",
                "./orion_config_bayes.yaml",
                "./rosenbrock.py",
                "-x~loguniform(1, 50, discrete=True)",
            ]
        )


def test_shape(monkeypatch):
    """Check support of multidim values."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    with OrionState(experiments=[], trials=[]):

        orion.core.cli.main(
            [
                "hunt",
                "--name",
                "exp",
                "--max-trials",
                "5",
                "--config",
                "./orion_config_bayes.yaml",
                "./rosenbrock.py",
                "-x~uniform(-5, 5, shape=3)",
            ]
        )


def test_bayesian_optimizer_two_inputs(monkeypatch):
    """Check functionality of BayesianOptimizer wrapper for 2 dimensions."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    with OrionState(experiments=[], trials=[]):

        orion.core.cli.main(
            [
                "hunt",
                "--config",
                "./orion_config_bayes.yaml",
                "./rosenbrock.py",
                "-x~uniform(-5, 5)",
                "-y~uniform(-10, 10)",
            ]
        )


def test_bayesian_optimizer_actually_optimize(monkeypatch):
    """Check if Bayesian Optimizer has better optimization than random search."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    best_random_search = 23.403275057472825

    with OrionState(experiments=[], trials=[]):

        orion.core.cli.main(
            [
                "hunt",
                "--name",
                "exp",
                "--max-trials",
                "20",
                "--config",
                "./orion_config_bayes.yaml",
                "./black_box.py",
                "-x~uniform(-50, 50, precision=10)",
            ]
        )

        exp = create_experiment(name="exp")

        objective = exp.stats["best_evaluation"]

        assert best_random_search > objective
