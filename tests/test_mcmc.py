import jax
import jax.numpy as jnp
import gradientdrift as gd
import pandas as pd

def get_toy_dataset(seed=42):
    """
    Generates a simple linear regression dataset.
    """
    key = jax.random.PRNGKey(seed)
    
    # Define "true" parameters
    true_intercept = 2.5
    true_slope = 1.3
    true_sigma = 0.5  # Noise level
    
    n_samples = 1000

    # Create the x-values
    x = jnp.linspace(-5, 5, n_samples)
    
    # Create the y-values
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, shape=(n_samples,)) * true_sigma
    y = true_intercept + true_slope * x + noise
    
    print(f"Toy Dataset: True intercept={true_intercept}, True slope={true_slope}, True sigma={true_sigma}")
    
    # This is the 'data' object you'll pass to your MCMC chunk
    data = {'x': x, 'y': y}
    return pd.DataFrame(data)

def test_mcmc_linear_regression():
    data = get_toy_dataset()

    trueParameters = {
        "intercept": 2.5,
        "slope": 1.3,
        "sigma": 0.5
    }

    formula = """
        {model.intercept} ~ normal(0, 10)
        {model.slope} ~ normal(0, 10)
        {model.sigma} ~ halfnormal(1)
        y ~ normal({model.intercept} + {model.slope} * x, {model.sigma})
    """

    data = gd.data.Dataset(data)
    model = gd.models.Universal(formula)
    model.fit(data, optimizer = "mcmc")
    model.summary(data, trueParams = trueParameters)

    allInConfidenceInterval = model.isAllInConfidenceInterval()
    assert allInConfidenceInterval, "Not all parameters are in the confidence interval"

if __name__ == "__main__":
    test_mcmc_linear_regression()
    print("MCMC linear regression test passed successfully!")