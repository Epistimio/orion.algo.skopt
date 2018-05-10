#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform integration tests for `orion.algo.skopt`."""
import os
import subprocess

import numpy
import pytest

from orion.core.worker.trial import Trial


@pytest.fixture(scope='session')
def database():
    """Return Mongo database object to test with example entries."""
    from pymongo import MongoClient
    client = MongoClient(username='user', password='pass', authSource='orion_test')
    database = client.orion_test
    yield database
    client.close()


@pytest.fixture()
def clean_db(database):
    """Clean insert example experiment entries to collections."""
    database.experiments.drop()
    database.trials.drop()
    database.workers.drop()
    database.resources.drop()


def present_results(database, exp_name):
    """Present 10 best trials in order from `exp_name`."""
    exp = list(database.experiments.find({'name': exp_name}))
    assert len(exp) == 1
    exp = exp[0]
    assert '_id' in exp
    exp_id = exp['_id']
    ctrials = list(database.trials.find({'experiment': exp_id, 'status': 'completed'}))
    losses = []
    trials = []
    for trial in ctrials:
        trial = Trial(**trial)
        trials.append(trial)
        losses.append(trial.objective.value)
    ctrials = trials
    losses = numpy.asarray(losses)
    idx = list(range(len(ctrials)))
    idx.sort(key=lambda x: losses[x])
    losses = losses[idx].tolist()
    ctrials[:] = [ctrials[i] for i in idx]
    loss_per_trial = zip(losses, ctrials)

    print("###           Top 10/{} Trials           ###".format(len(ctrials)))
    print("###    Loss  ---  Trial  ---  End Time    ###")
    for i, (loss, trial) in enumerate(loss_per_trial):
        if i == 10:
            break
        print(loss, trial, trial.end_time)


@pytest.mark.usefixtures("clean_db")
def test_bayesian_optimizer(database, monkeypatch):
    """Check functionality of BayesianOptimizer wrapper for single shaped dimension."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    process = subprocess.Popen(["orion", "--config", "./orion_config_bayes.yaml",
                                "./rosenbrock.py", "-x~uniform(-5, 5, shape=3)"])
    rcode = process.wait()
    assert rcode == 0
    present_results(database, "orion_skopt_bayes_test")


@pytest.mark.usefixtures("clean_db")
def test_bayesian_optimizer_two_inputs(database, monkeypatch):
    """Check functionality of BayesianOptimizer wrapper for 2 dimensions."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    process = subprocess.Popen(["orion", "--config", "./orion_config_bayes.yaml",
                                "./rosenbrock.py", "-x~uniform(-5, 5, shape=2)",
                                "-y~uniform(-10, 10)"])
    rcode = process.wait()
    assert rcode == 0
    present_results(database, "orion_skopt_bayes_test")
