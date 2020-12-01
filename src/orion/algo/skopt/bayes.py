# -*- coding: utf-8 -*-
"""
:mod:`orion.algo.skopt.bayes` -- Perform bayesian optimization
==============================================================

.. module:: bayes
   :platform: Unix
   :synopsis: Use Gaussian Process regression to locally search for a minimum.

"""
import logging

import numpy
from orion.algo.base import BaseAlgorithm
from orion.algo.space import check_random_state
from skopt import Optimizer, Space
from skopt.learning import GaussianProcessRegressor
from skopt.space import Real


log = logging.getLogger(__name__)


def orion_space_to_skopt_space(orion_space):
    """Convert Oríon's definition of problem's domain to a skopt compatible."""
    dimensions = []
    for key, dimension in orion_space.items():
        low, high = dimension.interval()
        shape = dimension.shape
        assert not shape or shape == [1]
        if not shape:
            shape = (1,)
            low = (low,)
            high = (high,)
        dimensions.append(Real(name=key, prior="uniform", low=low[0], high=high[0]))

    return Space(dimensions)


class BayesianOptimizer(BaseAlgorithm):
    """Wrapper skopt's bayesian optimizer"""

    requires_type = "real"
    requires_dist = "linear"
    requires_shape = "flattened"

    # pylint: disable = too-many-arguments
    def __init__(
        self,
        space,
        seed=None,
        strategy=None,
        n_initial_points=10,
        acq_func="gp_hedge",
        alpha=1e-10,
        n_restarts_optimizer=0,
        noise="gaussian",
        normalize_y=False,
    ):
        """Initialize skopt's BayesianOptimizer.

        Copying documentation from `skopt.learning.gaussian_process.gpr` and
        `skopt.optimizer.optimizer`.

        Parameters
        ----------
        space : `orion.algo.space.Space`
           Problem's definition
        seed: int (default: None)
           Seed used for the random number generator
        strategy : str (default: cl_min)
           Method to use to sample multiple points.
           Supported options are `"cl_min"`, `"cl_mean"` or `"cl_max"`.
           Check skopt docs for details.
        n_initial_points : int (default: 10)
           Number of evaluations of `func` with initialization points
           before approximating it with `base_estimator`. Points provided as
           `x0` count as initialization points. If len(x0) < n_initial_points
           additional points are sampled at random.
        acq_func : str (default: gp_hedge)
           Function to minimize over the posterior distribution. Can be:
           ``["LCB", "EI", "PI", "gp_hedge", "EIps", "PIps"]``. Check skopt
           docs for details.
        alpha : float or array-like (default: 1e-10)
           Value added to the diagonal of the kernel matrix during fitting.
           Larger values correspond to increased noise level in the observations
           and reduce potential numerical issue during fitting. If an array is
           passed, it must have the same number of entries as the data used for
           fitting and is used as datapoint-dependent noise level. Note that this
           is equivalent to adding a WhiteKernel with c=alpha. Allowing to specify
           the noise level directly as a parameter is mainly for convenience and
           for consistency with Ridge.
        n_restarts_optimizer : int (default: 0)
           The number of restarts of the optimizer for finding the kernel's
           parameters which maximize the log-marginal likelihood. The first run
           of the optimizer is performed from the kernel's initial parameters,
           the remaining ones (if any) from thetas sampled log-uniform randomly
           from the space of allowed theta-values. If greater than 0, all bounds
           must be finite. Note that n_restarts_optimizer == 0 implies that one
           run is performed.
        noise: str (default: "gaussian")
           If set to "gaussian", then it is assumed that y is a noisy estimate of f(x) where the
           noise is gaussian.
        normalize_y : bool (default: False)
           Whether the target values y are normalized, i.e., the mean of the
           observed target values become zero. This parameter should be set to
           True if the target values' mean is expected to differ considerable from
           zero. When enabled, the normalization effectively modifies the GP's
           prior based on the data, which contradicts the likelihood principle;
           normalization is thus disabled per default.

        .. seemore::
           About optional arguments passed to `skopt.learning.GaussianProcessRegressor`.

        """
        if strategy is not None:
            log.warning("Strategy is deprecated and will be removed in v0.1.2.")

        self._optimizer = None

        super(BayesianOptimizer, self).__init__(
            space,
            seed=seed,
            strategy=strategy,
            n_initial_points=n_initial_points,
            acq_func=acq_func,
            alpha=alpha,
            n_restarts_optimizer=n_restarts_optimizer,
            noise=noise,
            normalize_y=normalize_y,
        )

    @property
    def optimizer(self):
        """Return skopt's optimizer."""
        if self._optimizer is None:
            self._optimizer = Optimizer(
                base_estimator=GaussianProcessRegressor(
                    alpha=self.alpha,
                    n_restarts_optimizer=self.n_restarts_optimizer,
                    noise=self.noise,
                    normalize_y=self.normalize_y,
                ),
                dimensions=orion_space_to_skopt_space(self.space),
                n_initial_points=self.n_initial_points,
                acq_func=self.acq_func,
            )

            self.seed_rng(self.seed)

        return self._optimizer

    def seed_rng(self, seed):
        """Seed the state of the random number generator.

        :param seed: Integer seed for the random number generator.
        """
        self.seed = seed
        if self._optimizer:
            self.optimizer.rng.seed(seed)
            self.optimizer.base_estimator_.random_state = self.optimizer.rng.randint(
                0, 100000
            )

    @property
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the algorithm."""
        return {
            "optimizer_rng_state": self.optimizer.rng.get_state(),
            "estimator_rng_state": check_random_state(
                self.optimizer.base_estimator_.random_state
            ).get_state(),
        }

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        self.optimizer.rng.set_state(state_dict["optimizer_rng_state"])
        rng = numpy.random.RandomState(0)
        rng.set_state(state_dict["estimator_rng_state"])
        self.optimizer.base_estimator_.random_state = rng

    def suggest(self, num=1):
        """Suggest a `num`ber of new sets of parameters.

        Perform a step towards negative gradient and suggest that point.

        """
        if num > 1:
            raise AttributeError("BayesianOptimizer does not support num > 1.")

        return [self.optimizer._ask()]  # pylint: disable = protected-access

    def observe(self, points, results):
        """Observe evaluation `results` corresponding to list of `points` in
        space.

        Save current point and gradient corresponding to this point.

        """
        self.optimizer.tell(points, [r["objective"] for r in results])
