# Transductive Active Learning with Application to Safe Bayesian Optimization

This repository accompanies the Safe BO application from the paper ["Transductive Active Learning: Theory and Applications"](https://arxiv.org/abs/2402.15898).

The work was presented in an oral presentation at the ICML 2024 Workshop on Aligning Reinforcement Learning Experimentalists and Theorists. See the corresponding version of the paper [here](https://jonhue.github.io/assets/pdf/icml-2024-arlet.pdf).

## Getting started

### Installation

Requires Python 3.11.

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### CI checks

* The code is auto-formatted using `black .`.
* Static type checks can be run using `pyright`.
* Tests can be run using `python -m pytest tests`.

### Documentation

To start a local server hosting the documentation run ```PYTHONPATH=$(pwd) pdoc ./lib --math```.

### Reproducing the experiments

The `examples` directory contains python scripts that reproduce the experiments from the paper.
These examples simultaneously serve as examples of how to use the library in the `lib` directory.
