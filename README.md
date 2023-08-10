# FaDIn: Fast Discretized Inference For Hawkes Processes with General Parametric Kernels

![build](https://img.shields.io/github/actions/workflow/status/GuillaumeStaermanML/FaDIn/unit_tests.yml?event=push&style=for-the-badge)
![python version](https://img.shields.io/badge/python-3.7_|_3.8_|_3.9_|_3.10_|_3.11-blue?style=for-the-badge)
![license](https://img.shields.io/github/license/GuillaumeStaermanML/FaDIn?style=for-the-badge)
![code style](https://img.shields.io/badge/code_style-black-black?style=for-the-badge)

This Package implements FaDIn.

## Installation

**To install this package, make sure you have an up-to-date version of** `pip`.


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
```
