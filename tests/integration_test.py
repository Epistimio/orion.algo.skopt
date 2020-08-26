#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform integration tests for `orion.algo.skopt`."""
import os

import numpy
import pytest

from orion.algo.space import Integer, Real, Space
from orion.client import create_experiment
import orion.core.cli
from orion.core.worker.primary_algo import PrimaryAlgo
from orion.testing import OrionState


@pytest.fixture()
def space():
    """Return an optimization space"""
    space = Space()
    dim1 = Integer('yolo1', 'uniform', -3, 6)
    space.register(dim1)
    dim2 = Real('yolo2', 'uniform', 0, 1)
    space.register(dim2)

    return space


def test_seeding(space):
    """Verify that seeding makes sampling deterministic"""
    bayesian_optimizer = PrimaryAlgo(space, 'bayesianoptimizer')

    bayesian_optimizer.seed_rng(1)
    a = bayesian_optimizer.suggest(1)[0]
    assert not numpy.allclose(a, bayesian_optimizer.suggest(1)[0])

    bayesian_optimizer.seed_rng(1)
    assert numpy.allclose(a, bayesian_optimizer.suggest(1)[0])


def test_set_state(space):
    """Verify that resetting state makes sampling deterministic"""
    bayesian_optimizer = PrimaryAlgo(space, 'bayesianoptimizer')

    bayesian_optimizer.seed_rng(1)
    state = bayesian_optimizer.state_dict
    a = bayesian_optimizer.suggest(1)[0]
    assert not numpy.allclose(a, bayesian_optimizer.suggest(1)[0])

    bayesian_optimizer.set_state(state)
    assert numpy.allclose(a, bayesian_optimizer.suggest(1)[0])


def test_bayesian_optimizer_basic(monkeypatch):
    """Check functionality of BayesianOptimizer wrapper for single shaped dimension."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    with OrionState(experiments=[], trials=[]):

        orion.core.cli.main(["hunt", "--config",
                             "./orion_config_bayes.yaml", "./rosenbrock.py",
                             "-x~uniform(-5, 5)"])


def test_int(monkeypatch):
    """Check support of integer values."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    with OrionState(experiments=[], trials=[]):

        orion.core.cli.main(["hunt", "--name", "exp", "--max-trials", "5", "--config",
                             "./orion_config_bayes.yaml",
                             "./rosenbrock.py",
                             "-x~uniform(-5, 5, discrete=True)"])


def test_categorical(monkeypatch):
    """Check support of categorical values."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    with OrionState(experiments=[], trials=[]):

        orion.core.cli.main(["hunt", "--name", "exp", "--max-trials", "5", "--config",
                             "./orion_config_bayes.yaml",
                             "./rosenbrock.py",
                             "-x~choices([-5, -2, 0, 2, 5])"])


def test_bayesian_optimizer_two_inputs(monkeypatch):
    """Check functionality of BayesianOptimizer wrapper for 2 dimensions."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    with OrionState(experiments=[], trials=[]):

        orion.core.cli.main(["hunt", "--config",
                             "./orion_config_bayes.yaml", "./rosenbrock.py",
                             "-x~uniform(-5, 5)", "-y~uniform(-10, 10)"])


def test_bayesian_optimizer_actually_optimize(monkeypatch):
    """Check if Bayesian Optimizer has better optimization than random search."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    best_random_search = 23.403275057472825

    with OrionState(experiments=[], trials=[]):

        orion.core.cli.main(["hunt", "--name", "exp", "--max-trials", "20", "--config",
                             "./orion_config_bayes.yaml", "./black_box.py",
                             "-x~uniform(-50, 50, precision=10)"])

        exp = create_experiment(name='exp')

        objective = exp.stats['best_evaluation']

        assert best_random_search > objective
