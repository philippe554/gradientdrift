
import jax
import jax.numpy as jnp

def getRHat(allParams):
    def _per_leaf_r_hat(arr):
        n_chains = arr.shape[0]
        n_samples = arr.shape[1]
        
        # 1. Calculate per-chain mean
        # Shape: (n_chains, ...param_dims)
        chain_means = jnp.mean(arr, axis=1)

        # 2. Calculate per-chain variance (s_j^2)
        # Shape: (n_chains, ...param_dims)
        chain_vars = jnp.var(arr, axis=1, ddof=1)

        # 3. Calculate within-chain variance (W)
        # Shape: (...param_dims)
        W = jnp.mean(chain_vars, axis=0)

        # 4. Calculate between-chain variance (B)
        # Shape: (...param_dims)
        B = n_samples * jnp.var(chain_means, axis=0, ddof=1)

        # 5. Estimate target variance (V_hat)
        V_hat = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B

        # 6. Calculate R-hat
        # Add a small epsilon for numerical stability
        R_hat = jnp.sqrt(V_hat / (W + 1e-10))
        
        return R_hat

    return jax.tree_util.tree_map(_per_leaf_r_hat, allParams)