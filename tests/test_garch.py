import gradientdrift as gd 
from arch import arch_model

def equals(a, b, percentageError = 0.15):
    assert abs(a - b) <= percentageError * abs(b), f"Expected {b}, but got {a} (error: {abs(a - b)})"

def test_garch():

    formula = """
        {mu} = 0.05
        {omega}, {alpha} = 0.1
        {beta} = 0.85
        sigmaSq[0] = {omega} / (1 - {alpha} - {beta})
        sigmaSq = {omega} + {alpha} * (lag(y) - {mu})^2 + {beta} * lag(sigmaSq)
        y ~ normal({mu}, sqrt(sigmaSq))
    """

    generatorModel = gd.models.Universal(formula)
    states = generatorModel.predict(steps=10000)
    Y = states["y"]

    # Fit using external GARCH library as a test

    print("\nFitting external GARCH model...")

    garch_model = arch_model(Y, vol='garch', p=1, q=1)
    garch_fit = garch_model.fit(disp='off')
    print(garch_fit.summary())

    equals(garch_fit.params['mu'], 0.05)
    equals(garch_fit.params['omega'], 0.1)
    equals(garch_fit.params['alpha[1]'], 0.1)
    equals(garch_fit.params['beta[1]'], 0.85)

    # Fit using GradientDrift's Universal model

    print("\nFitting GradientDrift Universal model...")

    formula = """
        {mu} = 0
        {omega}, {alpha} = 0.01
        {beta} = 0.01
        sigmaSq[0] = {omega} / (1 - {alpha} - {beta})
        sigmaSq = {omega} + {alpha} * (lag(y) - {mu})^2 + {beta} * lag(sigmaSq)
        y ~ normal({mu}, sqrt(sigmaSq))
    """

    data = gd.data.Dataset(Y, ["y"])
    model = gd.models.Universal(formula)
    model.fit(data, optimizer = "lbfgs", maxNumberOfSteps = 50)
    model.summary(data)

    parameters = model.getConstraintParameters()
    equals(parameters["mu"], 0.05)
    equals(parameters["omega"], 0.1)
    equals(parameters["alpha"], 0.1)
    equals(parameters["beta"], 0.85)

    
