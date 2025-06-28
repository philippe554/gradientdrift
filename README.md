
# GradientDrift

[](https://www.google.com/search?q=https://pypi.org/project/gradientdrift/)
[](https://www.google.com/search?q=https://pypi.org/project/gradientdrift/)
[](https://www.google.com/search?q=https://github.com/philippe554/gradientdrift/actions/workflows/test.yml)
[](https://www.google.com/search?q=https://github.com/philippe554/gradientdrift/blob/main/LICENSE)

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

# Fit the model
model.fit(data, batchSize = 100)

# Optionally, provide the data to the summary function, this will calculate the confidence interval (but can be expensive)
model.summary(data)
```

## Contributing

Contributions are welcome\! Whether it's bug reports, feature requests, or new model implementations, your help is appreciated.

Please feel free to open an issue on the [GitHub Issue Tracker](https://github.com/philippe554/gradientdrift/issues) to start a discussion. If you plan to contribute code, please see the (forthcoming) `CONTRIBUTING.md` file for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=https://github.com/philippe554/gradientdrift/blob/main/LICENSE) file for details.