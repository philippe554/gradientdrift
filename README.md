
# GradientDrift

![PyPI Version](https://img.shields.io/pypi/v/gradientdrift)
![Python Versions](https://img.shields.io/pypi/pyversions/gradientdrift)
![Tests](https://github.com/philippe554/gradientdrift/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/pypi/l/gradientdrift)

GradientDrift is a Python library for high-performance econometric time series analysis, specifically designed for datasets that are too large to fit into memory. It leverages the power of [JAX](https://github.com/google/jax) for hardware acceleration (CPU/GPU/TPU) and just-in-time (JIT) compilation of models.

> ⚠️ **Under Active Development**
>
> This project is currently in a pre-release (beta) stage. The API may change in future versions, and while thoroughly tested, some model implementations should be considered experimental. We welcome feedback and contributions to stabilize and improve the library\!

-----

## Key Features

  * **Memory Efficient:** Processes data in batches, allowing you to train models on datasets of any size.
  * **High-Performance:** Uses JAX to JIT-compile and accelerate model fitting, making it significantly faster than traditional libraries for many workloads.
  * **Classic Models, Modern Backend:** Implements standard econometric models like VAR (Vector Autoregression) and GARCH with a modern, functional, and hardware-agnostic backend.
  * **Clean API:** A simple, intuitive interface for defining, fitting, and predicting with your models.

## Installation

You can install GradientDrift directly from PyPI:

```bash
pip install gradientdrift
```

## Quickstart

Here is a simple example of how to use GradientDrift to fit a VAR model on a sample dataset.

```python
import jax
import pandas as pd
import gradientdrift as gd 

numberOfLags = 1
numberOfVariables = 2

# The dataset object also accepts numpy arrays
data = gd.data.Dataset(pd.read_csv("test_data.csv"))

# Use any of the models, the fit and summary syntax bellow will be equivalent
model = gd.models.VAR(numberOfLags = numberOfLags, numberOfVariables = numberOfVariables)

# Fit the model (default uses ADAM optimizer)
model.fit(data, batchSize = 100)

# Optionally, provide the data to the summary function, this will calculate the confidence interval (but can be expensive)
model.summary(data)

# Alternative optimizers:
model.fit(data, optimizer="lbfgs")
model.fit(data, optimizer="ClosedForm")
```

A GARCH example where we first simulate data using a model specification:

```python
# Define the true parameters of a model
trueParams = {
    'mu': 0.05,
    'logOmega': jax.numpy.log(0.1),
    'logAlpha': jax.numpy.log(0.1),
    'logBeta': jax.numpy.log(0.85)
}

# Define a model
generatorModel = gd.models.GARCH(p=1, q=1)
generatorModel.setParameters(trueParams)
initialValues = generatorModel.getInitialValues()

# Simulate data
simulatedData = generatorModel.simulate(initialValues, steps=1000000)

# Put in a dataset container for batching
data = gd.data.Dataset(simulatedData)

# Fit the model
model = gd.models.GARCH(p = 1, q = 1)
model.fit(data)
model.summary(data, trueParams = trueParams) # Provide true params to show in the coefficient table
```

## Contributing

Contributions are welcome\! Whether it's bug reports, feature requests, or new model implementations, your help is appreciated.

Please feel free to open an issue on the [GitHub Issue Tracker](https://github.com/philippe554/gradientdrift/issues) to start a discussion. If you plan to contribute code, please see the (forthcoming) `CONTRIBUTING.md` file for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=https://github.com/philippe554/gradientdrift/blob/main/LICENSE) file for details.