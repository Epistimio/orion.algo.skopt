.. _scikit-bayesopt:

Scikit Bayesian Optimizer
-------------------------

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

.. autoclass:: orion.algo.skopt.bayes.BayesianOptimizer
   :exclude-members: space, state_dict, set_state, suggest, observe, is_done, seed_rng