# -*- coding: utf-8 -*-
"""
:mod:`orion.algo.skopt.bayes` -- Perform bayesian optimization
==============================================================

.. module:: bayes
   :platform: Unix
   :synopsis: Use Gaussian Process regression to locally search for a minimum.

"""
import numpy
from skopt import Optimizer, Space
from skopt.learning import GaussianProcessRegressor
from skopt.space import Real

from orion.algo.base import BaseAlgorithm
from orion.algo.space import (pack_point, unpack_point)


def convert_orion_space_to_skopt_space(orion_space):
    """Convert Oríon's definition of problem's domain to a skopt compatible."""
    dimensions = []
    for key, dimension in orion_space.items():
        #  low = dimension._args[0]
        #  high = low + dimension._args[1]
        low, high = dimension.interval()
        # NOTE: A hack, because orion priors have non-inclusive higher bound
        #       while scikit-optimizer have inclusive ones.
        high = numpy.nextafter(high, high - 1)
        shape = dimension.shape
        assert not shape or len(shape) == 1
        if not shape:
            shape = (1,)
        # Unpack dimension
        for i in range(shape[0]):
            dimensions.append(Real(name=key + '_' + str(i),
                                   prior='uniform',
                                   low=low, high=high))

    return Space(dimensions)


class BayesianOptimizer(BaseAlgorithm):
    """Wrapper skopt's bayesian optimizer"""

    requires = 'real'

    def __init__(self, space,
                 strategy='cl_min', n_initial_points=10, acq_func="gp_hedge",
                 alpha=1e-10, n_restarts_optimizer=0, normalize_y=False):
        """Initialize skopt's BayesianOptimizer.

        Copying documentation from `skopt.learning.gaussian_process.gpr` and
        `skopt.optimizer.optimizer`.

        Parameters
        ----------
        space : `orion.algo.space.Space`
           Problem's definition
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
        super(BayesianOptimizer, self).__init__(space,
                                                strategy=strategy,
                                                n_initial_points=n_initial_points,
                                                acq_func=acq_func,
                                                alpha=alpha,
                                                n_restarts_optimizer=n_restarts_optimizer,
                                                normalize_y=normalize_y)
        self.optimizer = None

    def suggest(self, num=1):
        """Suggest a `num`ber of new sets of parameters.

        Perform a step towards negative gradient and suggest that point.

        """
        self._init_optimizer()
        points = self.optimizer.ask(n_points=num, strategy=self.strategy)
        return [pack_point(point, self.space) for point in points]

    def observe(self, points, results):
        """Observe evaluation `results` corresponding to list of `points` in
        space.

        Save current point and gradient corresponding to this point.

        """
        self._init_optimizer()
        self.optimizer.tell([unpack_point(point, self.space) for point in points],
                            [r['objective'] for r in results])

    def _init_optimizer(self):
        if self.optimizer is None:
            self.optimizer = Optimizer(
                base_estimator=GaussianProcessRegressor(alpha=self.alpha,
                                                        n_restarts_optimizer=self.n_restarts_optimizer,
                                                        normalize_y=self.normalize_y),
                dimensions=convert_orion_space_to_skopt_space(self.space),
                n_initial_points=self.n_initial_points, acq_func=self.acq_func)
