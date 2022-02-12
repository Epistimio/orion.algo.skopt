.. _scikit-bayesopt:

Scikit Bayesian Optimizer
-------------------------

This algorithm class provides a wrapper for `Bayesian optimizer`_
using Gaussian process implemented in `scikit optimize`_.

.. _scikit optimize: https://scikit-optimize.github.io/
.. _bayesian optimizer: https://scikit-optimize.github.io/#skopt.Optimizer

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
                parallel_strategy:
                    of_type: StatusBasedParallelStrategy
                    strategy_configs:
                        broken:
                            of_type: MaxParallelStrategy
                    default_strategy:
                        of_type: NoParallelStrategy


.. autoclass:: orion.algo.skopt.bayes.BayesianOptimizer
   :exclude-members: space, state_dict, set_state, suggest, observe, is_done, seed_rng, get_data,
                     get_optimizer, get_y
