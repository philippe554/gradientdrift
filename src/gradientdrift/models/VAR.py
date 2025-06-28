
import jax

from .Model import Model

class VAR(Model):
    def __init__(self, numberOfLags=1, numberOfVariables=2):
        self.numberOfLags = numberOfLags
        self.numberOfVariables = numberOfVariables

    def requestPadding(self, dataset):
        dataset.setLeftPadding(self.numberOfLags)
        dataset.setRightPadding(0)

    def setParameters(self, params):
        if 'coeffs' not in params or 'const' not in params or 'logSigma' not in params:
            raise ValueError("Parameters must contain 'coeffs', 'const', and 'logSigma'.")

        if params['coeffs'].shape != (self.numberOfLags, self.numberOfVariables, self.numberOfVariables):
            raise ValueError(f"Coefficients must have shape ({self.numberOfLags}, {self.numberOfVariables}, {self.numberOfVariables}).")
        if params['const'].shape != (self.numberOfVariables,):
            raise ValueError(f"Constant must have shape ({self.numberOfVariables},).")
        if params['logSigma'].shape != (self.numberOfVariables,):
            raise ValueError(f"logSigma must have shape ({self.numberOfVariables},).")
        
        self.params = params

    def setRandomParameters(self, key):
        key, keyCoeffs, keyConst, keyLogSigma = jax.random.split(key, 4)

        self.params = {
            'coeffs': jax.random.normal(keyCoeffs, (self.numberOfLags, self.numberOfVariables, self.numberOfVariables)) * 0.1,
            'const': jax.random.normal(keyConst, (self.numberOfVariables,)) * 0.1, 
            'logSigma': jax.random.normal(keyLogSigma, (self.numberOfVariables,)) * 0.1
        }

    # x is of shape (batch size + left padding - 1, variables)
    def predict(self, params, x):
        batchSize = x.shape[0] - self.numberOfLags + 1
        
        yHat = jax.numpy.tile(params['const'], (batchSize, 1))

        for i in range(self.numberOfLags):
            coeffMatrix = params['coeffs'][i]

            start = self.numberOfLags - 1 - i
            end = start + batchSize
            laggedData = x[start:end, :]

            yHat += laggedData @ coeffMatrix

        return yHat
    
    def simulate(self, initialValues, steps, key):
        def loop_body(carry, _):
            last_p_values, currentKey = carry
            
            # Get the deterministic part of the prediction
            x_for_one_step = last_p_values
            y_hat_batch = self.predict(self.params, x_for_one_step) 
            y_hat_mean = y_hat_batch[0]
            
            # Create and add the random shock
            key, subkey = jax.random.split(currentKey)
            sigma = jax.nn.softplus(self.params['logSigma'])
            shock = jax.random.normal(subkey, shape=(self.numberOfVariables,)) * sigma
            y_simulated = y_hat_mean + shock

            # Update the state window
            new_carry = jax.numpy.vstack([last_p_values[1:], y_simulated])
            
            return (new_carry, key), y_simulated

        initial_carry = (initialValues, key)
        _, all_simulations = jax.lax.scan(loop_body, initial_carry, None, length=steps)
        return all_simulations

    def logLikelihood(self, params, data):
        x = data[:-1, :]
        y = data[self.numberOfLags:, :]

        if x.shape[0] != y.shape[0]:
            raise ValueError("Input x and y must have the same number of rows after padding.")

        yHat = self.predict(params, x)

        residuals = y - yHat

        sigma = jax.nn.softplus(params['logSigma'])
    
        log_prob = jax.scipy.stats.norm.logpdf(residuals, scale=sigma).sum(axis=1)
        
        return jax.numpy.mean(log_prob)
