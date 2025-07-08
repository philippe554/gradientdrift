import jax
import jax.numpy as jnp
import gradientdrift as gd 

# Define a model
generatorModel = gd.models.VARMAX(arLags = 3, maLags = 2, numberOfEndogVariables = 2)
#generatorModel = gd.models.VAR(numberOfLags = 3, numberOfVariables = 2)
trueParams = {
    'arCoeffs': jnp.array([
        # Lag 1: Strong positive autocorrelation
        [[0.6, 0.1],  # Eq1 depends on Var1(t-1) with 0.6, Var2(t-1) with 0.1
         [0.15, 0.5]], # Eq2 depends on Var1(t-1) with 0.15, Var2(t-1) with 0.5
        # Lag 2: Weaker positive autocorrelation
        [[0.2, 0.05],
         [0.05, 0.15]],
        # Lag 3: Some negative autocorrelation
        [[-0.1, 0.02],
         [0.02, -0.05]]
    ]),
    'maCoeffs': jnp.array([
        # Lag 1: Some mean reversion in the errors
        [[-0.3, 0.05],
         [0.05, -0.2]],
        # Lag 2: Weaker MA effect
        [[-0.1, 0.0],
         [0.0, -0.05]]
    ]),
    'const': jnp.array([0.5, 0.2]), # Gives the series a non-zero mean
    'logSigma': jnp.log(jnp.array([0.5, 0.5]))
}

generatorModel.setParameters(trueParams)

# Simulate data
simulatedData = generatorModel.simulate(steps=1000000)

# Put in a dataset container for batching
data = gd.data.Dataset(simulatedData)

# Fit the model
model = gd.models.VARMAX(arLags = 3, maLags = 2, numberOfEndogVariables = 2)
#model = gd.models.VAR(numberOfLags = 3, numberOfVariables = 2)
model.fit(data)
model.summary(data, trueParams = generatorModel.params) # Provide true params to show in the coefficient table
