import gradientdrift as gd 
import numpy as np
import pandas as pd

def generateData(trueParameters, observations = 1000):
    mu = trueParameters["mu"]
    alpha = trueParameters["alpha"]
    beta = trueParameters["beta"]

    np.random.seed(42)

    x1 = np.random.normal(0, 1, observations)
    x2 = np.random.normal(0, 1, observations)
    noise = np.random.normal(0, 1, observations)

    y = mu + alpha * x1 + beta * x2 + noise

    df = pd.DataFrame({
        "y": y,
        "x1": x1,
        "x2": x2
    })

    return df

def test_ols():

    trueParameters = {
        "mu": 0.05,
        "alpha": 0.1,
        "beta": 0.85
    }

    formula = """
        y ~ {mu} + {alpha} * x1 + {beta} * x2
    """
    
    df = generateData(trueParameters)

    data = gd.data.Dataset(df)
    model = gd.models.Universal(formula)
    model.fit(data)
    model.summary(data, trueParams = trueParameters)

    allInConfidenceInterval = model.isAllInConfidenceInterval()
    assert allInConfidenceInterval, "Not all parameters are in the confidence interval"

if __name__ == "__main__":
    test_ols()
    print("OLS test passed successfully!")