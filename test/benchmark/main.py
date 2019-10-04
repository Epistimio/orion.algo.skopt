import glob
import os
import matplotlib.pyplot as plt 


import orion.core.cli
from orion.core.io.experiment_builder import ExperimentBuilder


# train random, baye + current algo
# plot evolution...

database_config = { 
    "type": 'EphemeralDB'}


def order(trial):
    return trial.submit_time


def get_algorithm_configs():
    for file_name in glob.glob('*.yaml'):
        name, _ = os.path.splitext(file_name)
        yield name, file_name


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    for algo_name, algo_config_file in get_algorithm_configs():
        print(" ==== ")
        print(" Executing {}".format(algo_name))
        print(" ==== ")
        # experiment name based on file name
        orion.core.cli.main(
            ["hunt", "--config", algo_config_file, '-n', algo_name,
                "--max-trials", "40", "--pool-size", "1",
             "./rosenbrock.py", "-x~uniform(-10, 10, shape=2)", "-y~uniform(-10, 10)"])

    for algo_name, _ in get_algorithm_configs():

        experiment = ExperimentBuilder().build_view_from(
            {"name": algo_name, "database": database_config})

        objectives = []
        for trial in sorted(experiment.fetch_trials({'status': 'completed'}), key=order):
            objectives.append(min([trial.objective.value] + objectives))

        plt.plot(range(len(objectives)), objectives, label=algo_name)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
