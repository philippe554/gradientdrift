import gradientdrift as gd 
import numpy as np

def test_vecm():
    trueParameters = {
        "beta": 1.1765,
        "mu1": 0.005,
        "alpha1": -0.1,
        "gamma11": -0.1,
        "gamma12": 0.05,
        "mu2": 0.005,
        "alpha2": 0.2,
        "gamma21": 0.05,
        "gamma22": -0.15,
        "sigma1": 0.1,
        "sigma2": 0.1
    }

    formula = """
        ecm = lag(y1) - {beta} * lag(y2)

        y1[0] = 100
        y1[1] = 100
        y2[0] = 86
        y2[1] = 86

        Dy1 = {mu1} + {alpha1} * ecm + {gamma11} * lag(diff(y1)) + {gamma12} * lag(diff(y2))
        Dy2 = {mu2} + {alpha2} * ecm + {gamma21} * lag(diff(y1)) + {gamma22} * lag(diff(y2))

        Dy1[0] = 0
        Dy2[0] = 0
        Dy1[1] = 0
        Dy2[1] = 0
            
        diff(y1) ~ normal(Dy1, {sigma1})
        diff(y2) ~ normal(Dy2, {sigma2})
    """

    generatorModel = gd.models.Universal(formula)
    generatorModel.setParameters(trueParameters)
    states = generatorModel.predict(steps=10000)
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=PendingDeprecationWarning)
        from statsmodels.tsa.vector_ar.vecm import VECM
        print("Fitting VECM model using Statsmodels...")
        X = np.column_stack((states["y1"], states["y2"]))
        model = VECM(X, deterministic = "co")
        model_fit = model.fit()
        print(model_fit.summary()) 


    formula = """
        ErrorCorrectionModel:
            y1 ~ {beta} * y2
        VECM:
            y1[0] = 100
            y2[0] = 86
            y1[1] = 100
            y2[1] = 86

            diff(y1) ~ {mu1} + {alpha1} * (lag(data.y1) - {ErrorCorrectionModel.beta} * lag(data.y2)) + {gamma11} * lag(diff(y1)) + {gamma12} * lag(diff(y2))
            diff(y2) ~ {mu2} + {alpha2} * (lag(data.y1) - {ErrorCorrectionModel.beta} * lag(data.y2)) + {gamma21} * lag(diff(y1)) + {gamma22} * lag(diff(y2))
    """

    data = gd.data.Dataset(X, ["y1", "y2"])
    model = gd.models.Composite(formula)
    model.fit(data, optimizer = "lbfgs", maxNumberOfSteps = 100, burnInTime = 100)
    model.summary(data, all = True, trueParams = trueParameters)

    allInConfidenceInterval = model.isAllInConfidenceInterval()
    assert allInConfidenceInterval, "Not all parameters are in the confidence interval"

if __name__ == "__main__":
    test_vecm()
    print("VECM test passed successfully!")