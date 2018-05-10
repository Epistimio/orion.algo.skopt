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


@pytest.mark.usefixtures("clean_db")
def test_bayesian_optimizer(database, monkeypatch):
    """Check that random algorithm is used, when no algo is chosen explicitly."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    process = subprocess.Popen(["orion", "--config", "./orion_config_bayes.yaml",
                                "./rosenbrock.py", "-x~uniform(-10, 10, shape=3)"])
    rcode = process.wait()
    assert rcode == 0
    exp = list(database.experiments.find({'name': 'orion_skopt_bayes_test'}))
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

    print("###           Top 10/{} trials           ###".format(len(ctrials)))
    print("###    Loss  ---  Trial  ---  End Time    ###")
    for i, (loss, trial) in enumerate(loss_per_trial):
        if i == 10:
            break
        print(loss, trial, trial.end_time)
