# Second-Order Sensitivity Analysis for Bilevel Optimization

Experimental codebase accompanying the paper *Second-Order Sensitivity Analysis
for Bilevel Optimization* by Robert Dyro, Edward Schmerling, Nikos Arechiga and
Marco Pavone, published in *International Conference on Artificial Intelligence
and Statistics (AISTATS)*, 2022.

## Using this code

We do not recommend using this code directly because we maintain a separate
user-oriented version of this work implemented in
[PyTorch](https://pytorch.org/) and [JAX](https://github.com/google/jax).

You can find it here:
* [https://rdyro.github.io/sensitivity_jax/](https://rdyro.github.io/sensitivity_jax/)
* [https://rdyro.github.io/sensitivity_torch/](https://rdyro.github.io/sensitivity_torch/)

**We highly recommend navigating to those packages instead when interacting with
this code.**

## Citing this work

If you find this work useful, please cite this publication with the following
BibTeX
```
@inproceedings{DyroSchmerlingEtAl2022,
  author = {Dyro, R. and Schmerling, E. and Arechiga, N. and Pavone, M.},
  title = {Second-Order Sensitivity Analysis for Bilevel Optimization},
  booktitle = {{Int. Conf. on Artificial Intelligence and Statistics}},
  year = {2022},
  keywords = {press},
  owner = {rdyro},
  timestamp = {2022-02-05}
}
```

---

## Repository Organization

* `exps` contains the experiments used in the paper
  - `exps/svm` – Support Vector Machine hyperparameter tuning 
  - `exps/optimal_control` – inverse optimal control with constraints
  - `exps/auto_tuning` – model auto-tuning experiments
  - `exps/shared_scripts` – contains various general-purpose experimental
    scripts shared by the experiments

* `implicit` contains the main computational package
  - `implicit/interface.py` – interface bindings to make `jax` behave like
    `torch`
  - `implicit/implicit.py` – sensitivity routines and optimization function
    generation routine
  - `implicit/diff.py` – differentation utilities
  - `implicit/opt.py` – optimization routines
  - `implicit/nn_tools.py` – custom tools for working with neural networks in
    JAX
  - `implicit/pca.py` – principal component analysis visualization routines
  - `implicit/inverse.py` – specialized matrix-free inverse methods
  - `implicit/utils.py` – utility functions
* `unit_tests` contains some sanity checks and unit tests to verify the package
  is working as expected
