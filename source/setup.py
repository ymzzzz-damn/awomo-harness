# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

# One wheel contains the agent and its embodiment loader/descriptors.
setup(
    packages=find_packages("src") + ["embodiments", "embodiments.runtime"],
    package_dir={"": "src", "embodiments": "embodiments"},
)
