"""
This library implements algorithms for efficient unsupervised exploration of unknown (finite-domain) functions under (potentially also unknown) constraints using Gaussian processes.

There are two main components:
1. `lib.model` - A statistical Gaussian process model of the unknown functions.
1. `lib.algorithms` - The acquisition functions.

Additionally, `lib.function` wraps the unknown function and can be used to model the black-box functions of synthetic experiments.
`lib.gp` can be used to obtain a prior distribution by sampling some initial points.

The following pseudocode illustrates how model and algorithm can be used for black-box optimization.

```python
noise_rate: Noise = NOISE_ASSUMPTION
f: Function = UNKNOWN_FUNCTION
model: Model = INITIAL_MODEL
alg: Algorithm = ALGORITHM
for t in range(T):
    alg.step(f)
```
"""
