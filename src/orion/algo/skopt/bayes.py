# pylint: disable = no-name-in-module
# -*- coding: utf-8 -*-
"""
:mod:`orion.algo.skopt.bayes` -- Perform bayesian optimization
==============================================================

.. module:: bayes
   :platform: Unix
   :synopsis: Use Gaussian Process regression to locally search for a minimum.

"""
import contextlib
import copy
import logging
from collections import defaultdict

import numpy as np
from orion.algo.base import BaseAlgorithm
from orion.algo.parallel_strategy import strategy_factory
from orion.core.utils import format_trials
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
    parallel_strategy: dict or None, optional
        The configuration of a parallel strategy to use for pending trials or broken trials.
        Default is a MaxParallelStrategy for broken trials and NoParallelStrategy for pending
        trials.
    convergence_duplicates: int, optional
        Number of duplicate points the algorithm may sample before considering itself as done.
        Default: 10.

    """

    requires_type = "real"
    requires_dist = "linear"
    requires_shape = "flattened"

    # pylint: disable = too-many-arguments
    def __init__(
        self,
        space,
        seed=None,
        n_initial_points=10,
        acq_func="gp_hedge",
        alpha=1e-10,
        n_restarts_optimizer=0,
        noise="gaussian",
        normalize_y=False,
        parallel_strategy=None,
        convergence_duplicates=5,
    ):
        if parallel_strategy is None:
            parallel_strategy = {
                "of_type": "StatusBasedParallelStrategy",
                "strategy_configs": {
                    "broken": {
                        "of_type": "MaxParallelStrategy",
                    },
                },
                "default_strategy": {"of_type": "MaxParallelStrategy"},
            }

        self.strategy = strategy_factory.create(**parallel_strategy)

        self.rng = None
        self._optimizer_state = {}
        self._suggested = []

        super(BayesianOptimizer, self).__init__(
            space,
            seed=seed,
            n_initial_points=n_initial_points,
            acq_func=acq_func,
            alpha=alpha,
            n_restarts_optimizer=n_restarts_optimizer,
            noise=noise,
            normalize_y=normalize_y,
            parallel_strategy=parallel_strategy,
            convergence_duplicates=convergence_duplicates,
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

    @contextlib.contextmanager
    def get_optimizer(self):
        """Get resumed optimizer"""
        optimizer = Optimizer(
            base_estimator=GaussianProcessRegressor(
                alpha=self.alpha,
                n_restarts_optimizer=self.n_restarts_optimizer,
                noise=self.noise,
                normalize_y=self.normalize_y,
                random_state=self.rng.randint(0, np.iinfo(np.int32).max),
            ),
            random_state=self.rng,
            dimensions=orion_space_to_skopt_space(self.space),
            n_initial_points=self.n_initial_points,
            acq_func=self.acq_func,
            model_queue_size=1,
        )
        if "gains_" in self._optimizer_state:
            optimizer.gains_ = self._optimizer_state["gains_"]
        points, results = self.get_data()
        if points:
            optimizer.tell(points, results)

        yield optimizer

        # We keep gains_ to rebuild the Optimizer based on copy() method here:
        # https://github.com/scikit-optimize/scikit-optimize/blob/0.7.X/skopt/optimizer/optimizer.py#L272
        if hasattr(optimizer, "gains_"):
            self._optimizer_state["gains_"] = optimizer.gains_

    def seed_rng(self, seed):
        """Seed the state of the random number generator.

        :param seed: Integer seed for the random number generator.
        """
        if self.rng is None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng.seed(seed)

    @property
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the algorithm."""
        state_dict = copy.deepcopy(super(BayesianOptimizer, self).state_dict)

        state_dict["rng_state"] = copy.deepcopy(self.rng.get_state())
        state_dict["strategy"] = copy.deepcopy(self.strategy.state_dict)
        state_dict["_suggested"] = copy.deepcopy(self._suggested)
        state_dict["_optimizer_state"] = copy.deepcopy(self._optimizer_state)

        return state_dict

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        super(BayesianOptimizer, self).set_state(copy.deepcopy(state_dict))

        self.strategy.set_state(copy.deepcopy(state_dict["strategy"]))
        self.rng.set_state(copy.deepcopy(state_dict["rng_state"]))
        self._suggested = copy.deepcopy(state_dict["_suggested"])
        self._optimizer_state = copy.deepcopy(state_dict["_optimizer_state"])

    def suggest(self, num=None):
        """Suggest a `num`ber of new sets of parameters."""
        samples = []
        with self.get_optimizer() as optimizer:
            while len(samples) < num and not self.is_done:
                new_point = optimizer.ask()

                self._suggested.append(new_point)
                optimizer.tell(new_point, self.get_y(new_point))

                trial = format_trials.tuple_to_trial(new_point, self.space)

                if not self.has_suggested(trial):
                    self.register(trial)
                    samples.append(trial)

        return samples

    def get_data(self):
        """Get points with result or fake result if not completed"""
        points = copy.deepcopy(self._suggested)
        results = []
        for point in points:
            results.append(self.get_y(point))

        return points, results

    def get_y(self, point):
        """Get result or fake result if trial not completed"""
        trial = format_trials.tuple_to_trial(point, self.space)
        if self.has_observed(trial):
            return self._trials_info[self.get_id(trial)][0].objective.value

        return self.strategy.infer(trial).objective.value

    def observe(self, trials):
        """Observe evaluation `results` corresponding to list of `points` in
        space.

        """
        self.strategy.observe(trials)
        for trial in trials:
            self.register(trial)

    @property
    def is_done(self):
        """Whether the algorithm is done and will not make further suggestions.

        Return True, if an algorithm holds that there can be no further improvement.
        By default, the cardinality of the specified search space will be used to check
        if all possible sets of parameters has been tried.
        """
        hits = defaultdict(int)
        for point in self._suggested:
            hits[self.get_id(format_trials.tuple_to_trial(point, self.space))] += 1

        if hits and max(hits.values()) >= self.convergence_duplicates:
            return True

        if self.n_suggested >= self._original.cardinality:
            return True

        if self.n_suggested >= getattr(self, "max_trials", float("inf")):
            return True

        return False
