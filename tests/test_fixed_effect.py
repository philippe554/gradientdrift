import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import gradientdrift as gd

def test_fixed_effect():
    grunfeld_data = sm.datasets.grunfeld.load_pandas().data

    print("Fitting Fixed Effect Model (using OLS with dummies)...")
    fe_model = smf.ols('invest ~ value + capital + C(firm)', data=grunfeld_data).fit()
    print("Fixed Effect Model Summary:")
    print(fe_model.summary())

    trueParameters = {
        "alpha": fe_model.params["value"],
        "beta": fe_model.params["capital"],
        "fe": [
            fe_model.params["Intercept"],
            fe_model.params["Intercept"] + fe_model.params["C(firm)[T.Atlantic Refining]"],
            fe_model.params["Intercept"] + fe_model.params["C(firm)[T.Chrysler]"],
            fe_model.params["Intercept"] + fe_model.params["C(firm)[T.Diamond Match]"], 
            fe_model.params["Intercept"] + fe_model.params["C(firm)[T.General Electric]"],
            fe_model.params["Intercept"] + fe_model.params["C(firm)[T.General Motors]"],
            fe_model.params["Intercept"] + fe_model.params["C(firm)[T.Goodyear]"],
            fe_model.params["Intercept"] + fe_model.params["C(firm)[T.IBM]"],
            fe_model.params["Intercept"] + fe_model.params["C(firm)[T.US Steel]"],
            fe_model.params["Intercept"] + fe_model.params["C(firm)[T.Union Oil]"],
            fe_model.params["Intercept"] + fe_model.params["C(firm)[T.Westinghouse]"]
        ]
    }

    formula = "invest ~ {alpha} * value + {beta} * capital + {fe[firm]}"

    data = gd.data.Dataset(grunfeld_data)
    model = gd.models.Universal(formula)
    model.fit(data, batchSize = 55)
    model.summary(data, trueParams = trueParameters)

    allInConfidenceInterval = model.isAllInConfidenceInterval()
    assert allInConfidenceInterval, "Not all parameters are in the confidence interval"

    # print("Fitting Random Effect Model (using MixedLM)...")
    # re_model = smf.mixedlm('invest ~ value + capital', data=grunfeld_data, groups=grunfeld_data['firm']).fit()
    # print("Random Effect Model Summary:")
    # print(re_model.summary())

    # print("Fitting Random Effect Model (using MixedLM)...")
    # re_model = smf.mixedlm('invest ~ value + capital', data=grunfeld_data, groups=grunfeld_data['firm']).fit(reml = False)
    # print("Random Effect Model Summary:")
    # print(re_model.summary())

if __name__ == '__main__':
    test_fixed_effect()
