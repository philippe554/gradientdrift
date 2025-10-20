
import jax
import jax.numpy as jnp

def get_acf_1d(chain):
    """
    Calculate the autocorrelation function (ACF) of a 1D chain 
    using a Fast Fourier Transform (FFT).
    """
    chain = chain - jnp.mean(chain)
    n = chain.shape[0]
    
    # Pad with zeros to double the length
    f = jnp.fft.rfft(chain, n=2*n)
    
    # Compute the power spectral density
    power_spectrum = f * jnp.conj(f)
    
    # Inverse FFT gives the autocovariance
    autocov = jnp.fft.irfft(power_spectrum, n=2*n)[:n]
    
    # Normalize to get the autocorrelation
    return autocov / autocov[0]

def ess_1d(chain):
    """
    Calculate the Effective Sample Size (ESS) for a single 1D chain.
    
    This uses the "positive-pair" summing method from Stan/ArviZ
    to get a robust estimate of the integrated autocorrelation time.
    """
    n_samples = chain.shape[0]
    
    # 1. Get the autocorrelation function
    acf = get_acf_1d(chain)
    
    # 2. Sum positive pairs of autocorrelations
    # We sum (rho_1 + rho_2), (rho_3 + rho_4), ...
    # as long as the pair sum is positive.
    
    # Get all the pairs
    max_lag = n_samples // 2
    lag_indices = jnp.arange(1, max_lag)
    pairs = acf[2*lag_indices - 1] + acf[2*lag_indices]
    
    # Find the first index where the pair sum is negative
    # jnp.argmax(bool_array) returns the index of the first True
    first_neg_idx = jnp.argmax(pairs < 0)
    
    # Handle the case where all pairs are positive
    # (first_neg_idx will be 0, but pairs[0] will be > 0)
    sum_to_idx = jax.lax.cond(
        (first_neg_idx == 0) & (pairs[0] > 0),
        lambda: max_lag - 1,  # All positive, sum all pairs
        lambda: first_neg_idx   # Sum up to (but not including) this index
    )
    
    # Create a mask to sum only the positive pairs
    mask = jnp.arange(max_lag - 1) < sum_to_idx
    
    # 3. Calculate Integrated Autocorrelation Time (tau)
    # tau = 1 + 2 * sum(rho_t)
    # This is equivalent to: 1 + sum( (rho_{2t-1} + rho_{2t}) * 2 )
    tau = 1.0 + 2.0 * jnp.sum(pairs * mask)
    
    # 4. Calculate ESS
    # ESS can't be larger than the number of samples
    ess = jnp.minimum(n_samples, n_samples / tau)
    
    return ess

def calculate_ess(all_burned_in_samples):
    """
    Calculates the total Effective Sample Size (ESS) for a pytree 
    of burned-in samples.
    
    Assumes a single leaf 'arr' has shape (n_chains, n_samples, ...param_dims).
    
    It computes ESS for each chain and then sums them.
    """
    
    def _per_leaf_ess(arr):
        # arr shape: (n_chains, n_samples, ...param_dims)
        
        # 1. Reshape to (n_chains, n_samples, n_params_flat)
        n_chains, n_samples, *param_dims = arr.shape
        # Calculate flat parameter count, or 1 if it's a scalar
        n_params_flat = int(jnp.prod(jnp.array(param_dims))) if param_dims else 1
        arr_flat = arr.reshape((n_chains, n_samples, n_params_flat))
        
        # 2. We want to apply ess_1d to each chain (dim 0) and 
        # each flattened parameter (dim 2).
        # vmap over chains (in_axes=0)
        # vmap over params (in_axes=2, out_axes=1)
        vmapped_ess = jax.vmap(
            jax.vmap(ess_1d, in_axes=1, out_axes=0), 
            in_axes=0
        )
        
        # 3. Calculate ESS for each chain
        # ess_per_chain shape: (n_chains, n_params_flat)
        ess_per_chain = vmapped_ess(arr_flat)
        
        # 4. Total ESS is the sum of ESS from all chains
        # total_ess shape: (n_params_flat,)
        total_ess = jnp.sum(ess_per_chain, axis=0)
        
        # 5. Reshape back to original param_dims
        return total_ess.reshape(param_dims)

    # Apply this calculation to every leaf in the parameter pytree
    return jax.tree_util.tree_map(_per_leaf_ess, all_burned_in_samples)
