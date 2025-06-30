
import jax

from .model import Model

class VAR(Model):
    def __init__(self, numberOfLags=1, numberOfVariables=2):
        self.numberOfLags = numberOfLags
        self.numberOfVariables = numberOfVariables

        self.params = {
            'coeffs': jax.numpy.zeros((self.numberOfLags, self.numberOfVariables, self.numberOfVariables)),
            'const': jax.numpy.zeros((self.numberOfVariables,)),
            'logSigma': jax.numpy.zeros((self.numberOfVariables,))
        }

        self.paramDims = {
            'coeffs': ["L[i+1]", "VAR[i]", "EQ[i]"],
            'const': ["EQ[i]"],
            'logSigma': ["EQ[i]"]
        }

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

        yHat = self.predict(params, x)

        residuals = y - yHat

        sigma = jax.nn.softplus(params['logSigma'])
    
        log_prob = jax.scipy.stats.norm.logpdf(residuals, scale=sigma).sum(axis=1)
        
        return jax.numpy.mean(log_prob)

    def fitClosedForm(self, dataset, batchSize = 100):

        # TODO: Check transpositions of matrices
        # TODO: add JIT

        self.fitConfig = {
            "Method": "Closed Form",
            "Batch size": "Full batch" if batchSize == -1 else batchSize,
        }

        self.requestPadding(dataset)
        dataset.prepareBatches(batchSize)
        numberOfBatches = dataset.getNumberOfBatches()

        self.fitConfig['Number of batches'] = numberOfBatches

        numberIndependentVariables = self.numberOfLags * self.numberOfVariables + 1  # +1 for the constant term
        XXSum = jax.numpy.zeros((self.numberOfVariables, numberIndependentVariables, numberIndependentVariables))
        XYSum = jax.numpy.zeros((self.numberOfVariables, numberIndependentVariables, 1))

        for i in range(numberOfBatches):
            batch = dataset.getBatch(i)
            samplesInBatch = batch.data.shape[0] - self.numberOfLags

            # X is the same for every equation, so we can compute it once
            X = jax.numpy.ones((samplesInBatch, self.numberOfLags * self.numberOfVariables + 1))

            for lag in range(self.numberOfLags):
                X = X.at[:, 1 + lag * self.numberOfVariables : 1 + (1 + lag) * self.numberOfVariables].set(
                    batch.data[self.numberOfLags - 1 - lag:samplesInBatch + self.numberOfLags - 1 - lag, :])

            Y = batch.data[self.numberOfLags:, :].T.reshape(self.numberOfVariables, samplesInBatch, 1)

            XXSum += X.T @ X
            XYSum += X.T @ Y

        B = jax.numpy.linalg.solve(XXSum, XYSum)

        self.params['const'] = B[:, 0].reshape(self.numberOfVariables)
        self.params['coeffs'] = B[:, 1:].T.reshape(self.numberOfLags, self.numberOfVariables, self.numberOfVariables)

        errorSumOfSquares = jax.numpy.zeros((self.numberOfVariables, self.numberOfVariables))

        for i in range(numberOfBatches):
            batch = dataset.getBatch(i)
            x = batch.data[:-1, :]
            y = batch.data[self.numberOfLags:, :]
            yHat = self.predict(self.params, x)
            residuals = y - yHat
            errorSumOfSquares += residuals.T @ residuals

        sigma = jax.numpy.sqrt(jax.numpy.diag(errorSumOfSquares) / (dataset.getEffectiveNObs() - numberIndependentVariables))
        epsilon = 1e-9
        self.params['logSigma'] = jax.numpy.log(jax.numpy.exp(sigma) - 1 + epsilon)


