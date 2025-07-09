# FaDIn: a tool box for fast and robust inference for parametric point processes

![build](https://img.shields.io/github/actions/workflow/status/GuillaumeStaermanML/FaDIn/unit_tests.yml?event=push&style=for-the-badge)
![python version](https://img.shields.io/badge/python-3.7_|_3.8_|_3.9_|_3.10_|_3.11-blue?style=for-the-badge)
![license](https://img.shields.io/github/license/GuillaumeStaermanML/FaDIn?style=for-the-badge)
![code style](https://img.shields.io/badge/code_style-black-black?style=for-the-badge)

This package implements FaDIn and UNHaP. FaDIn and UNHaP are solvers for inference of Hawkes Processes with finite-support kernels on simulated or real data, with the following features:
- Computation time is low compared to other methods.
- Compatible in an univariate setting as well as a multivariate setting.
- Classical kernels (exponential truncated gaussian, raised cosine) are implemented. The user can add their own kernel for inference. 
- Masking: if only a few Hawkes Parameters need to be inferred, the user can mask the other parameters.
- Smart initialization of parameters before optimization: the user can choose between `random` (purely random), `moment_matching_max` (moment matching with maximum mode) and `moment_matching_mean` (moment matching with mean mode).


[FaDIn](https://proceedings.mlr.press/v202/staerman23a/staerman23a.pdf) does classical Hawkes inference with gradient descent.
[UNHaP](https://raw.githubusercontent.com/mlresearch/v258/main/assets/loison25a/loison25a.pdf) does Hawkes inference where the Hawkes Process is marked and mixed with a noisy Poisson process.


## Installation

**To install this package, make sure you have an up-to-date version of** `pip`.
```bash
python3 -m pip install --upgrade pip
```
### From PyPI (coming soon)

In a dedicated Python env, run:

```bash
pip install FaDIn
```

### From source

```bash
git clone https://github.com/mind-inria/FaDIn.git
cd FaDIn
```

In a dedicated Python env, run:

```bash
pip install -e .
```

Contributors should also install the development dependencies
in order to test and automatically format their contributions.

```bash
pip install -e ".[dev]"
pre-commit install
```

## Short examples
A few illustrative examples are provided in the `examples` folder of this repository, in particular:
- `plot_univariate_fadin`: simulate an univariate unmarked Hawkes process, infer Hawkes Process parameters using FaDIn, and plot inferred kernel.
- `plot_multivariate_fadin`: same as `plot_univariate_fadin` but in the multivariate case.
- `plot_unhap`: simulate an univariate marked Hawkes process and a marked Poisson process, infer Hawkes Process parameters using UNHaP, ald plot inferred kernels.

## Citing this work

If this package was useful to you, please cite it in your work:

```bibtex
@inproceedings{staerman2023fadin,
  title={Fadin: Fast discretized inference for hawkes processes with general parametric kernels},
  author={Staerman, Guillaume and Allain, C{\'e}dric and Gramfort, Alexandre and Moreau, Thomas},
  booktitle={International Conference on Machine Learning},
  pages={32575--32597},
  year={2023},
  organization={PMLR}
}

@improceedings{loison2025unhap,
title={UNHaP: Unmixing Noise from Hawkes Process},
author={Loison, Virginie and Staerman, Guillaume and Moreau, Thomas},
booktitle={International Conference on Artificial Intelligence and Statistics},
pages={1342--1350},
year={2025}
}
```
