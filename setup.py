#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Installation script for `orion.algo.skopt`."""
import os

from setuptools import setup

import versioneer


repo_root = os.path.dirname(os.path.abspath(__file__))

tests_require = ["pytest>=3.0.0"]

setup_args = dict(
    name="orion.algo.skopt",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Implement a wrapper for skopt optimizers.",
    long_description=open(
        os.path.join(repo_root, "README.rst"), "rt", encoding="utf8"
    ).read(),
    license="BSD-3-Clause",
    author=u"Epistímio",
    author_email="xavier.bouthillier@umontreal.ca",
    url="https://github.com/Epistimio/orion.algo.skopt",
    packages=["orion.algo.skopt"],
    package_dir={"": "src"},
    include_package_data=True,
    entry_points={
        "OptimizationAlgorithm": [
            "skopt_bayes = orion.algo.skopt.bayes:BayesianOptimizer"
        ],
    },
    install_requires=["orion>=0.1.15", "scikit-optimize>=0.5.1"],
    tests_require=tests_require,
    setup_requires=["setuptools", "pytest-runner>=2.0,<3dev"],
    extras_require=dict(test=tests_require),
    # "Zipped eggs don't play nicely with namespace packaging"
    # from https://github.com/pypa/sample-namespace-packages
    zip_safe=False,
)

setup_args["keywords"] = [
    "Machine Learning",
    "Deep Learning",
    "Distributed",
    "Optimization",
]

setup_args["platforms"] = ["Linux"]

setup_args["classifiers"] = [
    "Development Status :: 1 - Planning",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
] + [("Programming Language :: Python :: %s" % x) for x in "3 3.6 3.7 3.8 3.9".split()]

if __name__ == "__main__":
    setup(**setup_args)
