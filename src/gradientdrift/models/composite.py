
from gradientdrift.models import Universal
from gradientdrift.utils.formulaparsers import getParser
from gradientdrift.utils.formulawalkers import RemoveNewlines, AddNamespace
from lark import Tree
import jax

class Composite:
    def __init__(self, formula):
        self.formula = formula

        tokens = getParser("specification").parse(formula)
        tokens = RemoveNewlines().transform(tokens)

        self.models = []

        for spec in tokens.children:
            if type(spec) is Tree:
                if spec.data == "namedmodel":                   
                    name = spec.children[0].value
                    modelTokens = spec.children[1]
                    model = Universal(modelTokens)
                    self.models.append({
                        "name": name,
                        "obj": model
                    })

        self.mainModelIndex = len(self.models) - 1

    def constructModel(self, dataShape, key):
        assignments = []
        keys = jax.random.split(key, len(self.models))
        for i, model in enumerate(self.models):
            model["obj"].addAssignment(assignments)
            model["obj"].constructModel(dataShape, keys[i])
            assignments.extend(model["obj"].getAssignments(model["name"]))

    def fit(self, dataset, *args, **kwargs):
        constants = {}
        assignments = []
        for model in self.models:
            model.get("obj").setParameters(constants, "const")
            model["obj"].addAssignment(assignments)

            model["obj"].fit(dataset, *args, **kwargs)
            
            constants.update(model["obj"].getParameters(model["name"]))
            assignments.extend(model["obj"].getAssignments(model["name"]))

    def __getattr__(self, name):
        def dispatcher(*args, **kwargs):
            targetAll = kwargs.pop("all", False)
            targetName = kwargs.pop("name", None)
            
            if targetAll:
                results = {}
                for model in self.models:
                    func = getattr(model["obj"], name)
                    results[model["name"]] = func(*args, **kwargs)
                return results

            elif targetName:
                for model in self.models:
                    if model["name"] == targetName:
                        func = getattr(model["obj"], name)
                        return func(*args, **kwargs)
                raise AttributeError(f"No model found with name '{targetName}'")
            
            else:
                func = getattr(self.models[self.mainModelIndex]["obj"], name)
                return func(*args, **kwargs)

        return dispatcher

    def getModel(self, name):
        for model in self.models:
            if model["name"] == name:
                return model["obj"]
        raise ValueError(f"No model found with name '{name}'")
        