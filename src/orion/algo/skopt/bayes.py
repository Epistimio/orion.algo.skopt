# -*- coding: utf-8 -*-
"""
:mod:`orion.algo.skopt.bayes` -- Perform bayesian optimization
==================================================================================

.. module:: bayes
   :platform: Unix
   :synopsis: Use Gaussian Process regression to locally search for a minimum.

"""
import numpy

from skopt import Optimizer, Space
from skopt.learning import GaussianProcessRegressor
from skopt.space import Real, Integer, Categorical

from orion.algo.base import BaseAlgorithm


def convert_orion_space_to_skopt_space(orion_space):

    dimensions = []
    for key, dimension in orion_space.items():
        dimension_class = globals()[dimension.__class__.__name__]

        low = dimension._args[0]
        high = low + dimension._args[1]
        # NOTE: A hack, because orion priors have non-inclusive higher bound
        #       while scikit-optimizer have inclusive ones.
        high = high - numpy.abs(high - 0.0001) * 0.0001
        dimensions.append(
            dimension_class(
                name=key, prior=dimension._prior_name,
                low=low, high=high))

    return Space(dimensions)


class BayesianOptimizer(BaseAlgorithm):
    """Wrapper skopt's bayesian optimizer"""

    def __init__(self, space, **kwargs):
        super(BayesianOptimizer, self).__init__(space)

        self.optimizer = Optimizer(
            base_estimator=GaussianProcessRegressor(**kwargs),
            dimensions=convert_orion_space_to_skopt_space(space))

        self.strategy = "cl_min"

    def suggest(self, num=1):
        """Suggest a `num`ber of new sets of parameters.

        Perform a step towards negative gradient and suggest that point.

        """
        points = self.optimizer.ask(n_points=num, strategy=self.strategy)
        return points

    def observe(self, points, results):
        """Observe evaluation `results` corresponding to list of `points` in
        space.

        Save current point and gradient corresponding to this point.

        """
        self.optimizer.tell(points, [r['objective'] for r in results])

    @property
    def is_done(self):
        """Implement a terminating condition."""
        return False
