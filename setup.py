import os
from setuptools import setup

# borrowed from https://pythonhosted.org/an_example_pypi_project/setuptools.html


def read(fname):
    try:
        return open(os.path.join(os.path.dirname(__file__), fname)).read()
    except FileNotFoundError:
        return ""


setup(
    name="implicit",
    version="0.2.0",
    author="Robert Dyro",
    description=(
        "Code accompanying the paper 'Second-Order Sensitivity Analysis for Bilevel Optimization'"
    ),
    license="MIT",
    packages=["implicit"],
    long_description=read("README.md"),
)
