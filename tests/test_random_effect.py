import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import gradientdrift as gd
import pytest

@pytest.mark.skip(reason="Not implemented yet")
def test_random_effect():

    sleepstudy = sm.datasets.get_rdataset("sleepstudy", "lme4").data

    print(sleepstudy.head())

    model = smf.mixedlm("Reaction ~ Days", sleepstudy, groups=sleepstudy["Subject"]).fit()
    print(model.summary())

    formula = """
        Reaction ~ {beta_days} * Days + {re[Subject]}
    """

    data = gd.data.Dataset(sleepstudy, categoricalColumns = ["Subject"])
    model = gd.models.Universal(formula)
    model.fit(data)
    model.summary(data)

    formula = """
        {model.intercept} = [1]
        {model.beta_days} = [1]
        {model.sigma_re} = [1]
        {model.sigma_Reaction} = [1]
        {model.re} = [18]

        {model.intercept} ~ normal(250, 50)  # Prior centered near baseline
        {model.beta_days} ~ normal(10, 10)    # Prior for the slope
        {model.sigma_re} ~ halfnormal(50)     # Group variance
        {model.sigma_Reaction} ~ halfnormal(50)    # Observation noise
        {model.re[Subject]} ~ normal(0, {model.sigma_re})

        Reaction ~ normal({model.intercept} + {model.beta_days} * Days + {model.re[Subject]}, {model.sigma_Reaction})
    """

    data = gd.data.Dataset(sleepstudy, categoricalColumns = ["Subject"])
    model = gd.models.Universal(formula)
    model.fit(data, optimizer = "mcmc")
    model.summary(data)

    exit()

    grunfeld_data = sm.datasets.grunfeld.load_pandas().data

    print("Fitting Random Effect Model (using MixedLM)...")
    re_model = smf.mixedlm('invest ~ value + capital', data=grunfeld_data, groups=grunfeld_data['firm']).fit()
    print("Random Effect Model Summary:")
    print(re_model.summary())

    # Intercept  -54.031   26.624 -2.029 0.042 -106.213 -1.850
    # value        0.109    0.010 10.947 0.000    0.090  0.129
    # capital      0.308    0.016 18.803 0.000    0.276  0.340
    # Group Var 6738.611   62.760

    trueParameters = {
    }

    formula = """
        #{guide.intercept_loc} = -50
        {model.intercept} ~ normal(0, 100)
        
        {model.alpha} ~ normal(0, 10)
        {model.beta} ~ normal(0, 10)
        
        #{guide.stddev_scale} = 50
        
        #{guide.sigma_invest_scale} = 50
        {model.sigma_invest} ~ halfnormal(50)

        
        {model.stddev} ~ halfnormal(100)
        {model.re[firm]} ~ normal(0, {model.stddev})

        invest ~ {model.alpha} * value + {model.beta} * capital + {intercept} + {model.re[firm]}
    """

    data = gd.data.Dataset(grunfeld_data)
    model = gd.models.Universal(formula)
    model.fit(data, maxNumberOfSteps = 1000, learningRate = 0.001)
    model.summary(data, trueParams = trueParameters)

    allInConfidenceInterval = model.isAllInConfidenceInterval()
    assert allInConfidenceInterval, "Not all parameters are in the confidence interval"

    

    # print("Fitting Random Effect Model (using MixedLM)...")
    # re_model = smf.mixedlm('invest ~ value + capital', data=grunfeld_data, groups=grunfeld_data['firm']).fit(reml = False)
    # print("Random Effect Model Summary:")
    # print(re_model.summary())

if __name__ == '__main__':
    test_random_effect()
