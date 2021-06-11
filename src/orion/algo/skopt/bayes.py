# pylint: disable = no-name-in-module
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
    """Wrapper skopt's bayesian optimizer

    Parameters
    ----------
    space : ``orion.algo.space.Space``
       Problem's definition
    seed: int (default: None)
       Seed used for the random number generator
    strategy : str (default: cl_min)
       Method to use to sample multiple points.
       Supported options are ``["cl_min", "cl_mean", "cl_max"]``.
       Check skopt docs for details.
    n_initial_points : int (default: 10)
       Number of evaluations of trials with initialization points
       before approximating it with `base_estimator`. Points provided as
       ``x0`` count as initialization points. If ``len(x0) < n_initial_points``
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

    """

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
        if strategy is not None:
            log.warning("Strategy is deprecated and will be removed in v0.1.2.")

        self.optimizer = None

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
    def space(self):
        """Return transformed space of the BO"""
        return self._space

    @space.setter
    def space(self, space):
        """Set the space of the BO and initialize it"""
        self._original = self._space
        self._space = space
        self._initialize()

    def _initialize(self):
        """Initialize the optimizer once the space is transformed"""
        self.optimizer = Optimizer(
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

    def seed_rng(self, seed):
        """Seed the state of the random number generator.

        :param seed: Integer seed for the random number generator.
        """
        if self.optimizer:
            self.optimizer.rng.seed(seed)
            self.optimizer.base_estimator_.random_state = self.optimizer.rng.randint(
                0, 100000
            )

    @property
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the algorithm."""
        state_dict = super(BayesianOptimizer, self).state_dict

        if self.optimizer is None:
            return state_dict

        state_dict.update(
            {
                "optimizer_rng_state": self.optimizer.rng.get_state(),
                "estimator_rng_state": check_random_state(
                    self.optimizer.base_estimator_.random_state
                ).get_state(),
                "Xi": self.optimizer.Xi,
                "yi": self.optimizer.yi,
                # pylint: disable = protected-access
                "_n_initial_points": self.optimizer._n_initial_points,
                "gains_": getattr(self.optimizer, "gains_", None),
                "models": self.optimizer.models,
                "_next_x": getattr(self.optimizer, "_next_x", None),
            }
        )

        return state_dict

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        super(BayesianOptimizer, self).set_state(state_dict)
        if self.optimizer and "optimizer_rng_state" in state_dict:
            self.optimizer.rng.set_state(state_dict["optimizer_rng_state"])
            rng = numpy.random.RandomState(0)
            rng.set_state(state_dict["estimator_rng_state"])
            self.optimizer.base_estimator_.random_state = rng
            self.optimizer.Xi = state_dict["Xi"]
            self.optimizer.yi = state_dict["yi"]
            # pylint: disable = protected-access
            self.optimizer._n_initial_points = state_dict["_n_initial_points"]
            self.optimizer.gains_ = state_dict["gains_"]
            self.optimizer.models = state_dict["models"]
            # pylint: disable = protected-access
            self.optimizer._next_x = state_dict["_next_x"]

    def suggest(self, num=None):
        """Suggest a `num`ber of new sets of parameters.

        Perform a step towards negative gradient and suggest that point.

        """
        num = min(num, max(self.n_initial_points - self.n_suggested, 1))

        samples = []
        candidates = []
        while len(samples) < num:
            if candidates:
                candidate = candidates.pop(0)
                if candidate:
                    self.register(candidate)
                    samples.append(candidate)
            elif self.n_observed < self.n_initial_points:
                candidates = self._suggest_random(num)
            else:
                candidates = self._suggest_bo(max(num - len(samples), 0))

            if not candidates:
                break

        return samples

    def _suggest(self, num, function):
        points = []

        attempts = 0
        max_attempts = 100
        while len(points) < num and attempts < max_attempts:
            for candidate in function(num - len(points)):
                if not self.has_suggested(candidate):
                    self.register(candidate)
                    points.append(candidate)

                if self.is_done:
                    return points

            attempts += 1
            print(attempts)

        return points

    def _suggest_random(self, num):
        def sample(num):
            return self.space.sample(
                num, seed=tuple(self.optimizer.rng.randint(0, 1000000, size=3))
            )

        return self._suggest(num, sample)

    def _suggest_bo(self, num):
        # pylint: disable = unused-argument
        def suggest_bo(num):
            # pylint: disable = protected-access
            point = self.optimizer._ask()

            # If already suggested, give corresponding result to BO to sample another point
            if self.has_suggested(point):
                result = self._trials_info[self.get_id(point)][1]
                if result is None:
                    results = []
                    for _, other_result in self._trials_info.values():
                        if other_result is not None:
                            results.append(other_result["objective"])
                    result = numpy.array(results).mean()
                else:
                    result = result["objective"]

                self.optimizer.tell([point], [result])
                return []

            return [point]

        return self._suggest(num, suggest_bo)

    def observe(self, points, results):
        """Observe evaluation `results` corresponding to list of `points` in
        space.

        Save current point and gradient corresponding to this point.

        """
        to_tell = [[], []]
        for point, result in zip(points, results):
            if not self.has_observed(point):
                self.register(point, result)
                to_tell[0].append(point)
                to_tell[1].append(result["objective"])

        if to_tell[0]:
            self.optimizer.tell(*to_tell)

    @property
    def is_done(self):
        """Whether the algorithm is done and will not make further suggestions.

        Return True, if an algorithm holds that there can be no further improvement.
        By default, the cardinality of the specified search space will be used to check
        if all possible sets of parameters has been tried.
        """
        if self.n_suggested >= self._original.cardinality:
            return True

        if self.n_suggested >= getattr(self, "max_trials", float("inf")):
            return True

        return False
