
import numpy as np
import pandas as pd
import gradientdrift as gd
import pytest

def generate_ppca_data(N=100):
    """
    Generates a toy dataset based on a PPCA model.

    Generative process:
    z ~ Normal(0, I_K)
    y|z ~ Normal(W @ z + mu, sigma^2 * I_D)

    Args:
        N (int): The number of samples to generate.

    Returns:
        pd.DataFrame: A DataFrame with N rows and columns 'y1', 'y2', 'y3'.
    """
    
    # --- 1. Define Model Parameters ---
    
    # D = 3 (observed dimensions: y1, y2, y3)
    # K = 2 (latent dimensions: z1, z2)

    # W (Loading Matrix): Shape (D, K) = (3, 2)
    # These are arbitrary values chosen to create correlated data.
    W = np.array([
        [2.5, 0.5],  # y1 strongly influenced by z1
        [1.0, -1.5], # y2 influenced by z1 and -z2
        [-0.5, 2.0]  # y3 strongly influenced by z2
    ])

    # mu (Mean vector): Shape (D,) = (3,)
    # This is the mean for each observed variable.
    mu = np.array([5.0, 10.0, 0.0])

    # sigma (Noise standard deviation): scalar
    # This controls how much noise is added to the observations.
    sigma = 0.5

    # --- 2. Generate Latent Variables (z) ---
    
    # z ~ Normal(0, 1)
    # Shape (N, K) = (100, 2)
    K = W.shape[1]
    z = np.random.randn(N, K)

    # --- 3. Generate Observed Variables (y) ---

    # Calculate the mean component: W @ z.T
    # We need to be careful with shapes here.
    # z is (N, K). W is (D, K).
    # We want the result to be (N, D).
    # So we calculate: z @ W.T
    # (N, K) @ (K, D) -> (N, D)
    mean_component = z @ W.T

    # Add the mean vector mu
    # (N, D) + (D,) -> (N, D) (due to broadcasting)
    mean_y = mean_component + mu

    # Generate noise: Normal(0, sigma)
    # Shape (N, D)
    D = W.shape[0]
    noise = np.random.normal(loc=0.0, scale=sigma, size=(N, D))

    # Final observed data: y = mean_y + noise
    y = mean_y + noise

    # --- 4. Create pandas DataFrame ---
    
    df = pd.DataFrame(y, columns=['y1', 'y2', 'y3'])
    
    return df, W, mu, sigma

@pytest.mark.skip(reason="Not implemented yet")
def test_em():
    data, W, mu, sigma = generate_ppca_data()

    formula = """
        {W} = [2, 3]
        {mu} = [3]

        latent {z} = [100, 2]
        {z} ~ Normal(0, 1)

        y1, y2, y3 ~ normal( {W} @ {z} + {mu}, {sigma} )
    """

    data = gd.data.Dataset(data)
    model = gd.models.Universal(formula)
    model.fit(data, optimizer = "em")
    model.summary(data)

    allInConfidenceInterval = model.isAllInConfidenceInterval()
    assert allInConfidenceInterval, "Not all parameters are in the confidence interval"


if __name__ == "__main__":
    test_em()
    print("PPCA EM test passed successfully!")

"""
    theta = {W, mu, sigma}

    rhs = f(theta, z, x) --> x are data inputs (if any)

    argmax_theta [ log p(y | theta) ] --> intractable

    EM algorithm:

        theta_new = argmax_{theta_new} [ Q(theta_new | theta_old) ]

        Q(theta_new | theta_old) = E_{z ~ p(z | y, theta_old)} [ log p(y, z | theta_new) ]

            log p(y, z | theta) = log p(y | z, theta) + log p(z | theta) --> chain rule of probability

                log p(y | z, theta) = logpdf(data.y, loc = rhs, scale = sigma)

                log p(z | theta) = logpdf(z, loc = 0, scale = 1)

            log p(y, z | theta) = logpdf(data.y, loc = rhs, scale = sigma) + logpdf(z, loc = 0, scale = 1)

        Q(theta_new | theta_old)
          = E_{z ~ p(z | y, theta_old)} [
                logpdf(data.y, loc = rhs, scale = sigma_new) + logpdf(z, loc = 0, scale = 1)
            ]
          = E_{z ~ p(z | y, theta_old)} [
                logpdf(data.y, loc = rhs, scale = sigma_new)
            ] + E_{z ~ p(z | y, theta_old)} [
                logpdf(z, loc = 0, scale = 1)
            ]

        --> Use SymPy to symbolically get the sufficient statistics for the expectations

        If correct, I will need to the following sufficient statistics: E_z[z] and E_z[z @ z.T]


        ====

1. Search which sufficient statistics are required to calc Q using SymPy

2. Calculate them per sample using JAX Hessian and Jacobian of the likelihood and prior.

3. Insert them in Q, it should be solvable now

4. Check if Q has a closed form solution

5a. Solve using the closed form solution

5b. Solve using GD
           

"""