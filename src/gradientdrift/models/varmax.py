
# V = vector
# AR = autoregressive
# MA = moving average
# X = exogenous

import jax
import jax.numpy as jnp
from jax import lax

from .model import Model

class VARMAX(Model):
    """
    A Vector Autoregressive Moving Average model with Exogenous variables (VARMAX),
    implemented in JAX.

    This class serves as a general-purpose engine for various time series models,
    including VAR, VMA, VARMA, AR, MA, ARMA, and their counterparts with
    exogenous variables (VARX, ARIMAX, etc.).

    The model is defined by the equation for the endogenous variables y_t:
    y_t = const + SUM(ar_coeffs_i * y_{t-i}) 
              + SUM(exog_coeffs_k * x_{t-k}) 
              + SUM(ma_coeffs_j * residual_{t-j}) 
              + residual_t
    """

    def __init__(self, arLags, maLags, numberOfEndogVariables, numberOfExogVariables=0, exogLags=None, burnIn=100):
        """
        Initializes the VARMAX model structure.

        Args:
            arLags (list[int] or int): A list of positive integers for the AR lags, or an int for p.
            maLags (list[int] or int): A list of positive integers for the MA lags, or an int for q.
            numberOfEndogVariables (int): The number of endogenous time series (the y variables).
            numberOfExogVariables (int): The number of exogenous time series (the x variables).
            exogLags (list[int] or int, optional): A list of non-negative integers for the exogenous lags.
                                                  A lag of 0 is contemporaneous. Defaults to [0] if variables > 0.
            burnIn (int): The number of initial samples in a batch to discard from the likelihood
                          calculation to allow MA states to stabilize.
        """
        # --- Model Specification ---
        if isinstance(arLags, int):
            arLags = list(range(1, arLags + 1))
        if isinstance(maLags, int):
            maLags = list(range(1, maLags + 1))
        
        if numberOfExogVariables > 0 and exogLags is None:
            exogLags = [0] # Default to contemporaneous if not specified
        elif numberOfExogVariables == 0:
            exogLags = [] # No exog lags if no exog variables
        if isinstance(exogLags, int):
            exogLags = list(range(exogLags + 1))

        self.arLags = sorted(list(set(arLags)))
        self.maLags = sorted(list(set(maLags)))
        self.exogLags = sorted(list(set(exogLags)))
        self.numberOfEndogVariables = numberOfEndogVariables
        self.numberOfExogVariables = numberOfExogVariables
        self.burnIn = burnIn

        self.maxArLag = max(self.arLags) if self.arLags else 0
        self.maxMaLag = max(self.maLags) if self.maLags else 0
        self.maxExogLag = max(self.exogLags) if self.exogLags else -1 # Lag 0 is valid

        # The required padding is the max history needed for any variable.
        self.requiredPadding = max(self.maxArLag, self.maxExogLag if self.maxExogLag > -1 else 0)

        # --- Initialize Parameter Structures ---
        self.params = {}
        self.paramDims = {}

        self.params['const'] = jnp.zeros((self.numberOfEndogVariables,))
        self.paramDims['const'] = ["EQ[i]"]

        if self.arLags:
            self.params['arCoeffs'] = jnp.zeros((len(self.arLags), self.numberOfEndogVariables, self.numberOfEndogVariables))
            self.paramDims['arCoeffs'] = ["L[i]", "AR[i]", "EQ[i]"]

        if self.maLags:
            self.params['maCoeffs'] = jnp.zeros((len(self.maLags), self.numberOfEndogVariables, self.numberOfEndogVariables))
            self.paramDims['maCoeffs'] = ["L[i]", "MA[i]", "EQ[i]"]

        if self.numberOfExogVariables > 0:
            self.params['exogCoeffs'] = jnp.zeros((len(self.exogLags), self.numberOfExogVariables, self.numberOfEndogVariables))
            self.paramDims['exogCoeffs'] = ["L[i]", "EXOG[i]", "EQ[i]"]

        self.params['logSigma'] = jnp.zeros((self.numberOfEndogVariables,))
        self.paramDims['logSigma'] = ["EQ[i]"]

    def requestPadding(self, dataset):
        dataset.setLeftPadding(self.requiredPadding + self.burnIn)
        dataset.setRightPadding(0)

    def setParameters(self, params):
        """
        Validates and sets the parameters for the model.

        Args:
            params (dict): A dictionary containing the model parameters.
        """
        expected_keys = set(self.params.keys())
        provided_keys = set(params.keys())

        if expected_keys != provided_keys:
            missing = expected_keys - provided_keys
            extra = provided_keys - expected_keys
            message = ""
            if missing:
                message += f"Missing parameters: {missing}. "
            if extra:
                message += f"Unexpected parameters provided: {extra}."
            raise ValueError(message)

        # Validate shapes for each parameter
        if self.arLags:
            expected_shape = (len(self.arLags), self.numberOfEndogVariables, self.numberOfEndogVariables)
            if params['arCoeffs'].shape != expected_shape:
                raise ValueError(f"arCoeffs must have shape {expected_shape}, but got {params['arCoeffs'].shape}.")
        
        if self.maLags:
            expected_shape = (len(self.maLags), self.numberOfEndogVariables, self.numberOfEndogVariables)
            if params['maCoeffs'].shape != expected_shape:
                raise ValueError(f"maCoeffs must have shape {expected_shape}, but got {params['maCoeffs'].shape}.")

        if self.numberOfExogVariables > 0:
            expected_shape = (len(self.exogLags), self.numberOfExogVariables, self.numberOfEndogVariables)
            if params['exogCoeffs'].shape != expected_shape:
                raise ValueError(f"exogCoeffs must have shape {expected_shape}, but got {params['exogCoeffs'].shape}.")

        if params['const'].shape != (self.numberOfEndogVariables,):
            raise ValueError(f"const must have shape ({self.numberOfEndogVariables},), but got {params['const'].shape}.")
        
        if params['logSigma'].shape != (self.numberOfEndogVariables,):
            raise ValueError(f"logSigma must have shape ({self.numberOfEndogVariables},), but got {params['logSigma'].shape}.")
            
        self.params = params
        return self

    def setRandomParameters(self, key):
        """Initializes all model parameters with small random values."""
        newParams = {}
        numSplits = len(self.params)
        keys = jax.random.split(key, numSplits)
        keyDict = {k: keys[i] for i, k in enumerate(self.params.keys())}
        
        newParams['const'] = jax.random.normal(keyDict['const'], (self.numberOfEndogVariables,)) * 0.1
        if self.arLags:
            newParams['arCoeffs'] = jax.random.normal(keyDict['arCoeffs'], (len(self.arLags), self.numberOfEndogVariables, self.numberOfEndogVariables)) * 0.1
        if self.maLags:
            newParams['maCoeffs'] = jax.random.normal(keyDict['maCoeffs'], (len(self.maLags), self.numberOfEndogVariables, self.numberOfEndogVariables)) * 0.1
        if self.numberOfExogVariables > 0:
            newParams['exogCoeffs'] = jax.random.normal(keyDict['exogCoeffs'], (len(self.exogLags), self.numberOfExogVariables, self.numberOfEndogVariables)) * 0.1
        newParams['logSigma'] = jax.random.normal(keyDict['logSigma'], (self.numberOfEndogVariables,)) * 0.1
        
        self.params = newParams
        return self

    def _calculateConditionalMean(self, params, endogWindow, exogWindow, pastResiduals):
        """Calculates the one-step-ahead conditional mean."""
        yHat = params['const']

        # 1. Add AR component
        if self.arLags:
            for i, lag in enumerate(self.arLags):
                laggedData = endogWindow[-lag]
                yHat += laggedData @ params['arCoeffs'][i]

        # 2. Add MA component
        if self.maLags:
            for i, lag in enumerate(self.maLags):
                laggedResidual = pastResiduals[-lag]
                yHat += laggedResidual @ params['maCoeffs'][i]

        # 3. Add Exogenous component
        if self.numberOfExogVariables > 0:
            for i, lag in enumerate(self.exogLags):
                # Window is indexed with current time t at the end.
                # So, a lag of `k` corresponds to index `-1-k`.
                laggedExogData = exogWindow[-1 - lag]
                yHat += laggedExogData @ params['exogCoeffs'][i]

        return yHat

    def logLikelihood(self, params, endogData, exogData=None):
        """
        Calculates the log-likelihood, discarding an initial burn-in period.

        Args:
            params (dict): The dictionary of model parameters.
            endogData (jax.numpy.ndarray): The batch of endogenous data, shape (T, numEndogVars).
            exogData (jax.numpy.ndarray, optional): The batch of exogenous data, shape (T, numExogVars).

        Returns:
            float: The mean log-likelihood per observation after the burn-in period.
        """
        initialResiduals = jnp.zeros((self.maxMaLag, self.numberOfEndogVariables))
        effectiveT = endogData.shape[0] - self.requiredPadding

        def scan_body(carry, t):
            pastResiduals = carry
            
            # Get data windows for this time step `t`
            endogWindow = lax.dynamic_slice(
                endogData, (t, 0), (self.requiredPadding, self.numberOfEndogVariables)
            )
            
            exogWindow = None
            if self.numberOfExogVariables > 0:
                # Slice a window for all required exogenous lags
                start_idx = t + self.requiredPadding - self.maxExogLag
                exogWindow = lax.dynamic_slice(
                    exogData, (start_idx, 0), (self.maxExogLag + 1, self.numberOfExogVariables)
                )

            yHat = self._calculateConditionalMean(params, endogWindow, exogWindow, pastResiduals)
            yActual = endogData[t + self.requiredPadding]
            residual = yActual - yHat

            # --- FIX: Only update the residual window if there are MA terms ---
            newCarry = pastResiduals
            if self.maLags:
                newCarry = jnp.roll(pastResiduals, shift=-1, axis=0)
                newCarry = newCarry.at[-1, :].set(residual)
            
            return newCarry, residual

        _, residuals = lax.scan(scan_body, initialResiduals, jnp.arange(effectiveT))

        # Discard the burn-in period from the residuals
        residualsForLoss = residuals[self.burnIn:]
        numObs = residualsForLoss.shape[0]
        
        sigma = jnp.exp(params['logSigma'])
        logProb = jax.scipy.stats.norm.logpdf(residualsForLoss, scale=sigma).sum()
        
        # Calculate the mean log-likelihood
        meanLogProb = jnp.where(numObs > 0, logProb / numObs, -jnp.inf)
        
        # Return the POSITIVE mean log-likelihood. The optimizer's loss function should handle negation.
        return meanLogProb

    def simulate(self, steps, key = jax.random.PRNGKey(0), initialEndogData=None, exogData=None, initialExogData=None):
        """
        Simulates future values from the model.

        Args:
            params (dict): The dictionary of model parameters.
            steps (int): The number of steps to simulate into the future.
            key (jax.random.PRNGKey): JAX random key for generating shocks.
            initialEndogData (jax.numpy.ndarray, optional): The last `maxArLag` values of the endogenous series.
                                                            Defaults to zeros if None.
            exogData (jax.numpy.ndarray, optional): Future values of exogenous variables, shape (steps, numExogVars).
            initialExogData (jax.numpy.ndarray, optional): The last `maxExogLag` values of the exogenous series.
                                                           Defaults to zeros if None.

        Returns:
            jax.numpy.ndarray: The simulated data, shape (steps, numEndogVars).
        """
        # Handle optional initial state data
        if initialEndogData is None:
            initialEndogData = jnp.zeros((self.maxArLag, self.numberOfEndogVariables))
        elif initialEndogData.shape != (self.maxArLag, self.numberOfEndogVariables):
            raise ValueError(f"initialEndogData must have shape ({self.maxArLag}, {self.numberOfEndogVariables}).")

        if self.numberOfExogVariables > 0:
            if exogData is None or exogData.shape[0] < steps:
                raise ValueError(f"Future exogenous data for {steps} steps is required.")
            
            if initialExogData is None:
                initialExogData = jnp.zeros((self.maxExogLag, self.numberOfExogVariables))
            elif initialExogData.shape != (self.maxExogLag, self.numberOfExogVariables):
                raise ValueError(f"initialExogData must have shape ({self.maxExogLag}, {self.numberOfExogVariables}).")
        
        # Combine initial and future exogenous data to form a continuous window for scanning
        fullExogHistory = None
        if self.numberOfExogVariables > 0:
             fullExogHistory = jnp.vstack([initialExogData, exogData])

        initialCarry = {
            "endogWindow": initialEndogData,
            "residualWindow": jnp.zeros((self.maxMaLag, self.numberOfEndogVariables)),
            "key": key
        }

        def loop_body(carry, t):
            currentExogWindow = None
            if self.numberOfExogVariables > 0:
                # The window ends at the current time step `t` relative to the start of the future exogData
                # We need to index into the combined history
                start_idx = t
                currentExogWindow = lax.dynamic_slice(
                    fullExogHistory, (start_idx, 0), (self.maxExogLag + 1, self.numberOfExogVariables)
                )

            yHat = self._calculateConditionalMean(
                self.params, carry["endogWindow"], currentExogWindow, carry["residualWindow"]
            )

            newKey, subkey = jax.random.split(carry["key"])
            sigma = jnp.exp(self.params['logSigma'])
            shock = jax.random.normal(subkey, shape=(self.numberOfEndogVariables,)) * sigma
            ySimulated = yHat + shock

            newEndogWindow = jnp.roll(carry["endogWindow"], shift=-1, axis=0).at[-1, :].set(ySimulated)
            
            # --- FIX: Only update the residual window if there are MA terms ---
            newResidualWindow = carry["residualWindow"]
            if self.maLags:
                newResidualWindow = jnp.roll(carry["residualWindow"], shift=-1, axis=0).at[-1, :].set(shock)

            newCarry = {
                "endogWindow": newEndogWindow,
                "residualWindow": newResidualWindow,
                "key": newKey
            }
            
            return newCarry, ySimulated

        _, allSimulations = lax.scan(loop_body, initialCarry, jnp.arange(steps))
        
        return allSimulations
