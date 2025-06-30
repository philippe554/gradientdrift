import jax

from .model import Model

class GARCH(Model):
    def __init__(self, p=1, q=1):
        if p != 1 or q != 1:
            raise NotImplementedError("Only GARCH(1,1) is currently supported.")
        self.p = p
        self.q = q
        self.params = None
        self.effective_nobs = 0

        self.params = {
            'mu': 0.0,
            'logOmega': jax.numpy.log(0.1),
            'logAlpha': jax.numpy.log(0.1),
            'logBeta': jax.numpy.log(0.8)
        }

        self.paramDims = {
            'mu': [],
            'logOmega': [],
            'logAlpha': [],
            'logBeta': []
        }

    def requestPadding(self, dataset):
        dataset.setLeftPadding(100)
        dataset.setRightPadding(0)

    def setParameters(self, params):
        if 'mu' not in params or 'logOmega' not in params or 'logAlpha' not in params or 'logBeta' not in params:
            raise ValueError("Parameters must contain 'mu', 'logOmega', 'logAlpha', and 'logBeta'.")
        
        self.params = params

    def setRandomParameters(self, key):
        """Initializes unconstrained log-parameters with random values."""
        self.params = {
            'mu': jax.random.normal(key) * 0.1,
            'logOmega': jax.numpy.log(0.1),
            'logAlpha': jax.numpy.log(0.1),
            'logBeta': jax.numpy.log(0.8)
        }

    def predict(self, params, x):
        """
        For a simple GARCH model, the conditional mean prediction is just mu.
        This function exists for API consistency. Forecasting volatility is more complex.
        """
        return jax.numpy.full_like(x, params['mu'])
    
    def getInitialValues(self):
        trueOmega = jax.numpy.exp(self.params['logOmega'])
        trueAlpha = jax.numpy.exp(self.params['logAlpha'])
        trueBeta = jax.numpy.exp(self.params['logBeta'])
        initialSigmaSq = trueOmega / (1 - trueAlpha - trueBeta)
        initialY = self.params['mu']

        initialValues = {
            'initialY': self.params['mu'],
            'initialSigmaSq': initialSigmaSq
        }

        return initialValues

    def simulate(self, initialValues, steps, key = jax.random.PRNGKey(0)):
        """Simulates data from the GARCH process by adding a random error term."""
        
        def loop_body(carry, _):
            prev_y, prev_sigma_sq, currentKey = carry
            
            mu = self.params['mu']
            omega = jax.nn.softplus(self.params['logOmega'])
            alpha = jax.nn.softplus(self.params['logAlpha'])
            beta = jax.nn.softplus(self.params['logBeta'])

            prev_a_sq = (prev_y - mu)**2
            current_sigma_sq = omega + alpha * prev_a_sq + beta * prev_sigma_sq
            
            key, subkey = jax.random.split(currentKey)
            shock = jax.random.normal(subkey) * jax.numpy.sqrt(current_sigma_sq)
            current_y = mu + shock
            
            new_carry = (current_y, current_sigma_sq, key)
            return new_carry, current_y

        initial_carry = (initialValues['initialY'], initialValues['initialSigmaSq'], key)
        _, all_simulations = jax.lax.scan(loop_body, initial_carry, None, length=steps)
        return all_simulations

    def logLikelihood(self, params, data):
        """Calculates the log-likelihood for a GARCH(1,1) model using scan."""
        self.effective_nobs = data.shape[0]
        
        mu = params['mu']
        omega = jax.nn.softplus(params['logOmega'])
        alpha = jax.nn.softplus(params['logAlpha'])
        beta = jax.nn.softplus(params['logBeta'])
        
        def volatility_step(carry, y_t):
            prev_a_sq, prev_sigma_sq = carry
            current_sigma_sq = omega + alpha * prev_a_sq + beta * prev_sigma_sq
            a_t = y_t - mu
            new_carry = (a_t**2, current_sigma_sq)
            return new_carry, current_sigma_sq

        # Initialize with unconditional variance
        uncond_var = omega / (1 - alpha - beta)
        initial_carry = (uncond_var, uncond_var)
        
        _, sigma_sq_series = jax.lax.scan(volatility_step, initial_carry, data)
        
        logLikelihoods = jax.scipy.stats.norm.logpdf(data, loc=mu, scale=jax.numpy.sqrt(sigma_sq_series))
        return jax.numpy.mean(logLikelihoods)