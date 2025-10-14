import gradientdrift as gd 
from arch import arch_model

def test_garch():

    trueParameters = {
        "mu": 0.05,
        "omega": 0.1,
        "alpha": 0.1,
        "beta": 0.85
    }

    formula = """
        sigmaSq[0] = {omega} / (1 - {alpha} - {beta})
        sigmaSq = {omega} + {alpha} * (lag(y) - {mu})^2 + {beta} * lag(sigmaSq)
        y ~ normal({mu}, sqrt(sigmaSq))
    """

    generatorModel = gd.models.Universal(formula)
    generatorModel.setParameters(trueParameters)
    states = generatorModel.predict(steps=10000)
    Y = states["y"]

    # Fit using external GARCH library as a test

    print("\nFitting external GARCH model...")

    garch_model = arch_model(Y, vol='garch', p=1, q=1)
    garch_fit = garch_model.fit(disp='off')
    print(garch_fit.summary())

    # Fit using GradientDrift's Universal model

    print("\nFitting GradientDrift Universal model...")

    formula = """
        {mu} = 0
        {omega}, {alpha} = 0.01
        {beta} = 0.01
        0 < {alpha}, {beta}, {omega}
        sigmaSq[0] = {omega} / (1 - {alpha} - {beta})
        sigmaSq = {omega} + {alpha} * (lag(y) - {mu})^2 + {beta} * lag(sigmaSq)
        y ~ normal({mu}, sqrt(sigmaSq))
    """

    data = gd.data.Dataset(Y, ["y"])
    model = gd.models.Universal(formula)
    model.fit(data, optimizer = "lbfgs", maxNumberOfSteps = 50, burnInTime = 100)
    model.summary(data, trueParams = trueParameters)

    allInConfidenceInterval = model.isAllInConfidenceInterval()
    assert allInConfidenceInterval, "Not all parameters are in the confidence interval"

if __name__ == "__main__":
    test_garch()
    print("GARCH test passed successfully!")