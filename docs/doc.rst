.. _scikit-bayesopt:

Scikit Bayesian Optimizer
-------------------------

``orion.algo.skopt`` provides a wrapper for `Bayesian optimizer`_ using Gaussian process implemented
in `scikit optimize`_.

.. _scikit optimize: https://scikit-optimize.github.io/
.. _bayesian optimizer: https://scikit-optimize.github.io/#skopt.Optimizer

Installation
~~~~~~~~~~~~

.. code-block:: sh

   pip install orion.algo.skopt

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

    experiment:
        algorithms:
            BayesianOptimizer:
                seed: null
                n_initial_points: 10
                acq_func: gp_hedge
                alpha: 1.0e-10
                n_restarts_optimizer: 0
                noise: "gaussian"
                normalize_y: False

``seed``

``n_initial_points``

Number of evaluations of ``func`` with initialization points
before approximating it with ``base_estimator``. Points provided as
``x0`` count as initialization points. If len(x0) < n_initial_points
additional points are sampled at random.

``acq_func``

Function to minimize over the posterior distribution. Can be:
``["LCB", "EI", "PI", "gp_hedge", "EIps", "PIps"]``. Check skopt
docs for details.

``alpha``

Value added to the diagonal of the kernel matrix during fitting.
Larger values correspond to increased noise level in the observations
and reduce potential numerical issues during fitting. If an array is
passed, it must have the same number of entries as the data used for
fitting and is used as datapoint-dependent noise level. Note that this
is equivalent to adding a WhiteKernel with c=alpha. Allowing to specify
the noise level directly as a parameter is mainly for convenience and
for consistency with Ridge.

``n_restarts_optimizer``

The number of restarts of the optimizer for finding the kernel's
parameters which maximize the log-marginal likelihood. The first run
of the optimizer is performed from the kernel's initial parameters,
the remaining ones (if any) from thetas sampled log-uniform randomly
from the space of allowed theta-values. If greater than 0, all bounds
must be finite. Note that n_restarts_optimizer == 0 implies that one
run is performed.

``noise``

If set to "gaussian", then it is assumed that y is a noisy estimate of f(x) where the
noise is gaussian.

``normalize_y``

Whether the target values y are normalized, i.e., the mean of the
observed target values become zero. This parameter should be set to
True if the target values' mean is expected to differ considerable from
zero. When enabled, the normalization effectively modifies the GP's
prior based on the data, which contradicts the likelihood principle;
normalization is thus disabled per default.
