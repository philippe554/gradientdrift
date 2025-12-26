
import copy
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from lark import Tree

from gradientdrift.models.model import Model
from gradientdrift.utils.formulawalkers import *
from gradientdrift.utils.formulaparsers import getParser, ModelFormulaReconstructor
from gradientdrift.utils.functionmap import isDistributionFunction
from gradientdrift.utils.jaxhelpers import *
from gradientdrift.utils.sympyhelpers import *

def unpackPriors(priors, parameters, parameterizations):
    latentParameters = {}

    for prior in priors:
        lhs, rhs = prior["tokens"].children
        name = lhs.children[0].value
        if name in latentParameters:
            raise ValueError(f"Duplicate prior definition found for latent parameter '{name}'. Each latent parameter must have only one prior.")
        latentParameters[name] = {}

        if rhs.data == "funccall":
            distributionName = rhs.children[0].value
            args = rhs.children[1].children if len(rhs.children) > 1 else []
            if distributionName == "normal":
                if len(args) != 2:
                    raise ValueError(f"Normal distribution must have exactly two arguments: mean and standard deviation. Found: {len(args)} arguments.")
                
                nameWithoutNamespace = name.split(".")[-1]
                guideName = "guide." + nameWithoutNamespace

                # tokens to calculate the logpdf of the latent variable with respect to the prior
                latentParameters[name]["prior"] = Tree(Token("RULE", "funccall"), [
                    Token("FUNCNAME", "norm.logpdf"),
                    Tree(Token("RULE", "arguments"), [
                        Tree(Token("RULE", "parameter"), [Token("VALUENAME", name)]),
                        args[0],
                        args[1]
                    ])
                ])
                # tokens to go from the guide to the latent variable
                latentParameters[name]["sample"] = Tree(Token("RULE", "sum"), [
                    Tree(Token("RULE", "parameter"), [Token("VALUENAME", guideName + "_loc")]),
                    Tree("operator", [Token("SUMOPERATOR", "+")]),
                    Tree(Token("RULE", "product"), [
                        Tree(Token("RULE", "parameter"), [Token("VALUENAME", guideName + "_scale")]),
                        Tree("operator", [Token("PRODUCTOPERATOR", "*")]),
                        Tree(Token("RULE", "funccall"), [
                            Token("FUNCNAME", "normal"),
                            Tree(Token("RULE", "arguments"), [
                                Tree(Token("RULE", "number"), [Token("SIGNED_NUMBER", "0")]),
                                Tree(Token("RULE", "number"), [Token("SIGNED_NUMBER", "1")])
                            ])
                        ])
                    ])
                ])
                # tokens to calculate the logpdf of the latent variable with respect to the guide
                latentParameters[name]["guide"] = Tree(Token("RULE", "funccall"), [
                    Token("FUNCNAME", "norm.logpdf"),
                    Tree(Token("RULE", "arguments"), [
                        Tree(Token("RULE", "parameter"), [Token("VALUENAME", name)]),
                        Tree(Token("RULE", "parameter"), [Token("VALUENAME", guideName + "_loc")]),
                        Tree(Token("RULE", "parameter"), [Token("VALUENAME", guideName + "_scale")])
                    ])
                ])
                # tokens to generate a sample from the prior
                shape = parameters[name]["shape"] if "shape" in parameters[name] else (1,)
                shape = shape[0] if len(shape) == 1 else 1
                latentParameters[name]["generate"] = Tree(Token("RULE", "funccall"), [
                    Token("FUNCNAME", "normal"),
                    Tree(Token("RULE", "arguments"), [
                        args[0],
                        args[1],
                        Tree(Token("RULE", "array"), [
                            Tree(Token("RULE", "number"), [Token("SIGNED_NUMBER", shape)])
                        ])
                    ])
                ])

                size = 1
                if guideName == "guide.re":
                    size = 18

                if guideName + "_loc" not in parameters:
                    parameters[guideName + "_loc"] = {}
                parameters[guideName + "_loc"]["shape"] = (size,)
                # parameters[guideName + "_loc"]["tokens"] = args[0]

                if guideName + "_scale" not in parameters:
                    parameters[guideName + "_scale"] = {}
                parameters[guideName + "_scale"]["shape"] = (size,)
                # parameters[guideName + "_scale"]["tokens"] = args[1]
                parameterizations[guideName + "_scale"] = {
                    "unconstraintParameterNames": [guideName + "_scale"],
                    "apply": lambda p, v = 0: jnp.exp(p),
                    "inverse": lambda p, v = 0: jnp.log(p)
                }

            elif distributionName == "halfnormal":
                if len(args) != 1:
                    raise ValueError(f"Halfnormal distribution must have exactly one argument: standard deviation. Found: {len(args)} arguments.")
                
                nameWithoutNamespace = name.split(".")[-1]
                guideName = "guide." + nameWithoutNamespace

                # tokens to calculate the logpdf of the latent variable with respect to the prior
                latentParameters[name]["prior"] = Tree(Token("RULE", "sum"), [
                    Tree(Token("RULE", "funccall"), [
                        Token("FUNCNAME", "norm.logpdf"),
                        Tree(Token("RULE", "arguments"), [
                            Tree(Token("RULE", "parameter"), [Token("VALUENAME", name)]),
                            Tree(Token("RULE", "number"), [Token("SIGNED_NUMBER", "0")]),
                            args[0]
                        ])
                    ]),
                    Tree("operator", [Token("SUMOPERATOR", "+")]),
                    Tree(Token("RULE", "funccall"), [
                        Token("FUNCNAME", "log"),
                        Tree(Token("RULE", "arguments"), [
                            Tree(Token("RULE", "number"), [Token("SIGNED_NUMBER", "2")])
                        ])
                    ])
                ])
                # tokens to go from the guide to the latent variable
                latentParameters[name]["sample"] = Tree(Token("RULE", "product"), [
                    Tree(Token("RULE", "parameter"), [Token("VALUENAME", guideName + "_scale")]),
                    Tree("operator", [Token("PRODUCTOPERATOR", "*")]),
                    Tree(Token("RULE", "funccall"), [
                        Token("FUNCNAME", "abs"), 
                        Tree(Token("RULE", "arguments"), [
                            Tree(Token("RULE", "funccall"), [
                                Token("FUNCNAME", "normal"),
                                Tree(Token("RULE", "arguments"), [
                                    Tree(Token("RULE", "number"), [Token("SIGNED_NUMBER", "0")]),
                                    Tree(Token("RULE", "number"), [Token("SIGNED_NUMBER", "1")])
                                ])
                            ])
                        ])
                    ])
                ])
                # tokens to calculate the logpdf of the latent variable with respect to the guide
                latentParameters[name]["guide"] = Tree(Token("RULE", "sum"), [
                    Tree(Token("RULE", "funccall"), [
                        Token("FUNCNAME", "norm.logpdf"),
                        Tree(Token("RULE", "arguments"), [
                            Tree(Token("RULE", "parameter"), [Token("VALUENAME", name)]),
                            Tree(Token("RULE", "number"), [Token("SIGNED_NUMBER", "0")]),
                            Tree(Token("RULE", "parameter"), [Token("VALUENAME", guideName + "_scale")])
                        ])
                    ]),
                    Tree("operator", [Token("SUMOPERATOR", "+")]),
                    Tree(Token("RULE", "funccall"), [
                        Token("FUNCNAME", "log"),
                        Tree(Token("RULE", "arguments"), [
                            Tree(Token("RULE", "number"), [Token("SIGNED_NUMBER", "2")])
                        ])
                    ])
                ])
                # tokens to generate a sample from the prior
                latentParameters[name]["generate"] = Tree(Token("RULE", "funccall"), [
                    Token("FUNCNAME", "abs"),
                    Tree(Token("RULE", "arguments"), [
                        Tree(Token("RULE", "funccall"), [
                            Token("FUNCNAME", "normal"),
                            Tree(Token("RULE", "arguments"), [
                                Tree(Token("RULE", "number"), [Token("SIGNED_NUMBER", "0")]),
                                args[0]
                            ])
                        ])
                    ])
                ])

                if guideName + "_scale" not in parameters:
                    parameters[guideName + "_scale"] = {}

                parameters[guideName + "_scale"]["shape"] = (1,)
                parameterizations[guideName + "_scale"] = {
                    "unconstraintParameterNames": [guideName + "_scale"],
                    "apply": lambda p, v = 0: jnp.exp(p),
                    "inverse": lambda p, v = 0: jnp.log(p)
                }
                # parameters[guideName + "_scale"]["tokens"] = args[0]

                parameterizations[name] = {
                    "unconstraintParameterNames": [name],
                    "apply": lambda p, v = 0: jnp.exp(p),
                    "inverse": lambda p, v = 0: jnp.log(p)
                }


            else:
                raise ValueError(f"Unsupported distribution '{distributionName}' in the prior.")
        else:
            raise ValueError(f"Right-hand side of the prior must be a distribution function. Found: {rhs.data}.")
        
    return latentParameters

def insertParameters(formulaTree):
    # parameters = GetParameterNames().transform(rhs)

    # if len(parameters) == 0: # No parameters found, automatic mode
    #     LabelOuterSum().visit(rhs)
    #     rhs = InsertParameters().transform(rhs)
    #     parameters = GetParameterNames().transform(rhs)
    #     if len(parameters) == 0:
    #         raise ValueError("No parameters found in the formula. Please ensure the formula is correctly specified.")
    #     if len(set(parameters)) != len(parameters):
    #         raise ValueError("Duplicate parameters found in the formula. Please ensure each parameter is unique.")

    return formulaTree

def convertToProbabilisticFormula(formulaTree):
    lhs, rhs = formulaTree.children

    if rhs.data == "funccall" and isDistributionFunction(rhs.children[0].value):
        return formulaTree
    
    else:
        formulaName = getParameterName(lhs)
        sigmaName = formulaName + "_sigma"
        sigma = Tree(Token("RULE", "parameter"), [Token("VALUENAME", sigmaName)])

        rhs = insertParameters(rhs)

        tokens = Tree(Token("RULE", "funccall"), [
            Token("FUNCNAME", "normal"),
            Tree(Token("RULE", "arguments"), [rhs, sigma])
        ])
        return Tree(Token("RULE", "formula"), [lhs, tokens])

def isolateLHS(lhs, rhs, dataColumns, modelName):
    lhsVariables = GetVariables().transform(lhs)
    if len(set(lhsVariables)) != 1:
        raise ValueError(f"Left-hand side of the assignment must contain exactly one variable. Found: {', '.join(lhsVariables)}.")
    
    if lhs.data == "array":
        allChildrenAreVariables = all(child.data == "variable" for child in lhs.children)
        if allChildrenAreVariables:
            return copy.deepcopy(lhs), copy.deepcopy(rhs)
        else:
            if len(lhs.children) != 1:
                raise ValueError(f"A non-pure lhs can only have one child. Found: {len(lhs.children)} children.")
            else:
                return isolateLHS(lhs.children[0], rhs, dataColumns, modelName)

    elif lhs.data == "variable":
        return copy.deepcopy(lhs), copy.deepcopy(rhs)        

    elif lhs.data == "funccall" and lhs.children[0].value == "diff":
        lhsInterated = lhs.children[1].children[0]

        constantTerm = Tree(Token('RULE', 'variable'), [Token('VALUENAME', lhsInterated.children[0].value)])
        constantTerm = AddNamespace(dataColumns = dataColumns, modelNamespace = modelName).transform(constantTerm)

        rhsAssignment = Tree(Token("RULE", "sum"), [
            Tree(Token("RULE", "funccall"), [
                Token("FUNCNAME", "lag"),
                Tree(Token("RULE", "arguments"), [constantTerm])
            ]),
            Tree("operator", [
                Token("SUMOPERATOR", "+")
            ]),
            copy.deepcopy(rhs)
        ])

        return copy.deepcopy(lhsInterated), copy.deepcopy(rhsAssignment)
    else:
        raise ValueError(f"Left-hand side of the assignment can not be processed. Found: {lhs.data}.")

def searchClosedFormSolution(lossFormula):
    print("Searching for closed-form solution for the loss formula")

    larkToSymPy = LarkToSymPy()
    expr = larkToSymPy.transform(lossFormula)
    print("Loss expression:", expr)

    parameters = GetParameterNames().transform(lossFormula)
    parameters = ['mu', 'alpha', 'beta']  # Temporary hardcoded for testing
    print("Parameters found:", parameters)

    parameterSymbols = [sp.Symbol(p) for p in parameters]
    grads = [sp.diff(expr, p) for p in parameterSymbols]
    for p, g in zip(parameters, grads):
        print(f"Gradient w.r.t. {p}:", g)

    solution = sp.solve(grads, parameterSymbols)
    print("Closed-form solution found:", solution)
    

    exit()

class Universal(Model):
    def __init__(self, formula):
        self.formula = formula
        self.importedAssignments = []
        self.importedConstants = {}
        self.importedParameters = {}

    def constructModel(self, dataShape, key, modelName = "model"):
        if modelName == "model":
            print("Constructing model:")
        else:
            print(f"Constructing model '{modelName}':")

        self.modelName = modelName

        # Tokenize the formula
        if type(self.formula) is str:
            formulaTree = getParser().parse(self.formula)
        elif type(self.formula) is Tree:
            formulaTree = self.formula
        else:
            raise ValueError(f"Unsupported formula type: {type(self.formula)}. Expected str or Tree.")
        formulaTree = RemoveNewlines().transform(formulaTree)
        
        statements = formulaTree.children
        parameterDefinition = [{"tokens" : s} for s in statements if s.data == 'parameterdefinition']
        constraintDefinitions = [{"tokens" : s} for s in statements if s.data == 'constraint']
        self.initializations = [{"tokens" : s} for s in statements if s.data == 'initialization']
        self.assignments = [{"tokens" : s} for s in statements if s.data == 'assignment']
        self.formulas = [{"tokens" : s} for s in statements if s.data == 'formula']
        self.optimize = [{"tokens" : s} for s in statements if s.data == 'optimize']
        self.priors = [{"tokens" : s} for s in statements if s.data == 'prior']
        self.OLS = {}
        self.EM = []

        # Process parameter definitions (should come first)
        self.parameters = {}
        parameterDefinitionsFound = set()
        for i, definition in enumerate(parameterDefinition):
            names = []
            for name in definition["tokens"].children[0].children:
                name = name.children[0].value
                if "." in name:
                    if name.split(".")[0] == modelName or name.split(".")[0] == "guide":
                        names.append(name)
                    else:
                        raise ValueError(f"Parameter name '{name}' does not match the model name '{modelName}'. Please ensure the parameter names are correctly defined.")
                else:
                    names.append(modelName + "." + name)
            
            if any(name in parameterDefinitionsFound for name in names):
                raise ValueError(f"Duplicate parameter definition found for names: {', '.join(names)}. Each parameter must be defined only once.")
            parameterDefinitionsFound.update(names)
            
            for name in names:
                self.parameters[name] = {}
                for rhs in definition["tokens"].children[1:]:
                    if rhs.data == "shape": # Set the shape of the parameter
                        shape = tuple([int(e.value) for e in rhs.children])
                        self.parameters[name]["shape"] = shape
                        # print(f"Parameter '{name}' shape set to {shape}.")
                    else: # Set to a constant value. A sum can flatten to multiple token types, thus match anything
                        self.parameters[name]["tokens"] = rhs
                        # print(f"Parameter '{name}' initialized to {ModelFormulaReconstructor.reconstruct(copy.deepcopy(rhs))}.")

                if name == "model.z":
                    print("Latent variable 'z' detected.")
                    self.parameters[name]["isLatent"] = True

                    
        # Process constraints (order independent)
        self.parameterizations = {}
        for i, constraint in enumerate(constraintDefinitions):
            # Upgrade the constraint tokens to include the model namespace
            constraint["tokens"] = AddNamespace([], modelNamespace = self.modelName).transform(constraint["tokens"])

            if constraint["tokens"].children[0].data == "boundconstraint":
                constraintTokens = constraint["tokens"].children[0].children
                if len(constraintTokens) == 2:
                    left = constraintTokens[0]
                    right = constraintTokens[1]

                    valueGetter = GetValueFromData()
                    if right.data == "parameterlist":
                        value = valueGetter.transform(left)
                        parameters = GetParameterNames().transform(right)
                        for parameter in parameters:
                            self.parameterizations[parameter] = {
                                "unconstraintParameterNames": [parameter],
                                "apply": lambda p, v = value: jax.nn.softplus(p) + v,
                                "inverse": lambda p, v = value: jnp.log(jnp.exp(p - v) - 1)
                            }
                    else:
                        raise ValueError(f"Right-hand side of bound constraint must be a parameter list. Found: {right.data}.")
                else:
                    raise ValueError(f"Bound constraint must have exactly 3 children: left-hand side, operator, and right-hand side. Found {len(constraintTokens)} children.")
            else:
                raise ValueError(f"Unsupported constraint type '{constraint['tokens'].data}'. Supported types are 'boundconstraint' and 'sumconstraint'.")

        # Process initializations (order independent, evaluated in the forward pass)
        for i, initialization in enumerate(self.initializations):
            initialization["name"] = initialization["tokens"].children[0].value
            if "." in initialization["name"]:
                if initialization["name"].split(".")[0] != modelName:
                    raise ValueError(f"Initialization name '{initialization['name']}' does not match the model name '{modelName}'. Please ensure the initialization names are correctly defined.")
            else:
                initialization["name"] = modelName + "." + initialization["name"]

            initialization["index"] = int(initialization["tokens"].children[1].value)
            rhs = initialization["tokens"].children[2]
            rhs = AddNamespace(dataColumns = {}, modelNamespace = self.modelName).transform(rhs)
            initialization["value"] = rhs

            # print(f"Initialization '{initialization['name']}' at index {initialization['index']} set to {ModelFormulaReconstructor.reconstruct(copy.deepcopy(initialization['value']))}.")

        # Process formulas (insert parameters and split into assignments and optimizations)
        for i, formula in enumerate(self.formulas):
            lhs, rhs = convertToProbabilisticFormula(formula["tokens"]).children

            dependingVariables = GetVariables().transform(lhs)
            for var in dependingVariables:
                if "." in var:
                    raise ValueError(f"LHS of a formula must not contain namespaces.")

            lhsParameters = GetParameterNames().transform(lhs)
            if len(lhsParameters) != 0:
                raise ValueError(f"Left-hand side of the assignment must not contain parameters. Found: {', '.join(lhsParameters)}.")
                
            if lhs.data != "array":
                raise ValueError(f"LHS of a formula must be an array of variables. Found: {lhs.data}.")
                
            # Name unnamed parameters
            parameters = GetParameterNames().transform(rhs)
            rhs = NameUnnamedParameters(parameters).transform(rhs)
            
            distributionName = rhs.children[0].value
            if distributionName == "normal":
                args = rhs.children[1].children
                if len(args) != 2:
                    raise ValueError(f"Normal distribution must have exactly two arguments: mean and standard deviation. Found: {len(args)} arguments.")

                lhsAssignment, rhsAssignment = isolateLHS(lhs, rhs, dataShape.keys(), self.modelName)
                assignmentTokens = Tree(Token("RULE", "assignment"), [lhsAssignment, rhsAssignment])
                self.assignments.append({
                    "tokens": assignmentTokens
                })
            
                lhsData = AddNamespace(dataColumns = dataShape.keys(), modelNamespace = self.modelName).transform(lhs)
                args = rhs.children[1].children
                rhsMean = args[0]
                rhsStd = args[1]

                loss = Tree(Token("RULE", "funccall"), [
                    Token("FUNCNAME", "norm.logpdf"),
                    Tree(Token("RULE", "arguments"), [
                        lhsData,
                        rhsMean,
                        rhsStd
                    ])
                ])

                tokens = Tree(Token("RULE", "optimize"), [
                    Token("OPTIMIZETYPE", "maximize"),
                    loss
                ])
                self.optimize.append({"tokens": tokens})

                searchClosedFormSolution(loss)

                # self.EM.append({
                #     "variable": lhsData,
                #     "mean": rhsMean,
                #     "std": rhsStd
                # })
            else:
                raise ValueError(f"Unsupported distribution '{distributionName}' in the formula. Supported distributions are: normal, uniform, poisson, binomial, exponential, gamma, beta, cauchy, laplace, lognormal.")
            
            # else:
            #     lhs = lhs.children[0] # TODO: Change this when the formula parser supports multiple left-hand sides

            #     ### Copy from bellow

            #     if lhs.data != "variable":
            #         lhsParameters = GetParameterNames().transform(lhs)
            #         if len(lhsParameters) != 0:
            #             raise ValueError(f"Left-hand side of the assignment must not contain parameters. Found: {', '.join(lhsParameters)}.")
            #         lhsVariables = GetVariables().transform(lhs)
            #         if len(set(lhsVariables)) != 1:
            #             raise ValueError(f"Left-hand side of the assignment must contain exactly one variable. Found: {', '.join(lhsVariables)}.")

            #         if lhs.data == "funccall" and lhs.children[0].value == "diff":
            #             lhs = lhs.children[1].children[0]

            #             # lhs will be on the rhs, so remove the namespace and add it again based on data columns available
            #             #constantTerm = copy.deepcopy(lhs)
            #             #constantTerm.children[0].value = constantTerm.children[0].value.replace("model.", "") # TODO
            #             constantTerm = Tree(Token('RULE', 'variable'), [Token('VALUENAME', lhs.children[0].value)])
            #             constantTerm = AddNamespace(dataColumns = dataShape.keys(), modelNamespace = self.modelName).transform(constantTerm)

            #             rhs = Tree(Token("RULE", "sum"), [
            #                 Tree(Token("RULE", "funccall"), [
            #                     Token("FUNCNAME", "lag"),
            #                     Tree(Token("RULE", "arguments"), [constantTerm])
            #                 ]),
            #                 Tree("operator", [
            #                     Token("SUMOPERATOR", "+")
            #                 ]),
            #                 copy.deepcopy(rhs)
            #             ])
            #         else:
            #             raise ValueError(f"Left-hand side of the assignment must be a variable or a diff. Found: {lhs.data}.")


            #     ###

            #     dependingVariables = GetVariables().transform(lhs)
            #     if len(dependingVariables) != 1:
            #         raise ValueError(f"Left-hand side of the formula must contain exactly one variable. Found: {', '.join(dependingVariables)}.")
            #     dependingVariable = dependingVariables[0]

            #     lhsModel = AddNamespace(dataColumns = [], modelNamespace = self.modelName).transform(lhs)
            #     lhsData = AddNamespace(dataColumns = dataShape.keys(), modelNamespace = self.modelName).transform(lhs)
            #     rhs = AddNamespace(dataColumns = dataShape.keys(), modelNamespace = self.modelName).transform(rhs)
                
            #     rhs = ExpandFunctions().transform(rhs)
            #     SetLeafNodeLags().visit(rhs)
                
            #     rhs = LabelDataDependencies().transform(rhs)

            #     tokens = Tree(Token("RULE", "assignment"), [lhsModel, rhs])
            #     self.assignments.append({"tokens": tokens})

            #     if rhs.meta.dataDependency in ["linearTerm", "linearSum", "linearSumAndConstant"]:
            #         # TODO: move this to a separate function
            #         OLSTermsGetter = GetOLSTerms(dataShape)
            #         OLSTermsGetter.visit(rhs)
            #         self.OLS[dependingVariable] = {
            #             "terms": OLSTermsGetter.terms,
            #             "bias": OLSTermsGetter.bias,
            #             "biasIsPositive": OLSTermsGetter.biasIsPositive,
            #             "constants": OLSTermsGetter.constants
            #         }

            #         for term in self.OLS[dependingVariable]["terms"]:
            #             print(f"    OLS term: {term} * {ModelFormulaReconstructor.reconstruct(copy.deepcopy(self.OLS[dependingVariable]['terms'][term]))}")
            #         if self.OLS[dependingVariable]["bias"] is not None:
            #             print(f"    OLS bias: {self.OLS[dependingVariable]['bias']}")
            #         for constant in self.OLS[dependingVariable]["constants"]:
            #             print(f"    OLS constant: {ModelFormulaReconstructor.reconstruct(copy.deepcopy(constant))}")

            #     tokens = Tree(Token("RULE", "optimize"), [
            #         Token("OPTIMIZETYPE", "maximize"),
            #         Tree(Token("RULE", "funccall"), [
            #             Token("FUNCNAME", "norm.logpdf"),
            #             Tree(Token("RULE", "arguments"), [
            #                 lhsModel,
            #                 lhsData,
            #                 Tree(Token("RULE", "parameter"), [
            #                     Token("VALUENAME", "model.sigma_" + dependingVariable)
            #                 ])
            #             ])
            #         ])
            #     ])
            #     self.optimize.append({"tokens": tokens})

            #     self.parameterizations["model.sigma_" + dependingVariable] = {
            #         "unconstraintParameterNames": ["model.sigma_" + dependingVariable],
            #         "apply": lambda p, v = 0: jnp.exp(p),
            #         "inverse": lambda p, v = 0: jnp.log(p)
            #     }

        for importedAssignment in self.importedAssignments:
            print(f"    Imported assignment: {importedAssignment['name']} = {ModelFormulaReconstructor.reconstruct(copy.deepcopy(importedAssignment['rhs']))}")

        # Process assignments (should come after formulas)
        for i, assignment in enumerate(self.assignments):
            lhs, rhs = assignment["tokens"].children
            del assignment["tokens"]

            lhs = AddNamespace(dataColumns = [], modelNamespace = self.modelName).transform(lhs)
            rhs = AddNamespace(dataColumns = dataShape.keys(), modelNamespace = self.modelName).transform(rhs)

            rhs = ExpandFunctions().transform(rhs)

            if lhs.data == "variable":
                assignment["name"] = lhs.children[0].value
            else:
                lhsParameters = GetParameterNames().transform(lhs)
                if len(lhsParameters) != 0:
                    raise ValueError(f"Left-hand side of the assignment must not contain parameters. Found: {', '.join(lhsParameters)}.")
                lhsVariables = GetVariables().transform(lhs)
                if len(set(lhsVariables)) != 1:
                    raise ValueError(f"Left-hand side of the assignment must contain exactly one variable. Found: {', '.join(lhsVariables)}.")
                assignment["name"] = lhsVariables[0]

            #     if lhs.data == "funccall" and lhs.children[0].value == "diff":
            #         lhs = lhs.children[1].children[0] 
                    
            #         # lhs will be on the rhs, so remove the namespace and add it again based on data columns available
            #         constantTerm = copy.deepcopy(lhs)
            #         constantTerm.children[0].value = constantTerm.children[0].value.replace("model.", "")
            #         constantTerm = AddNamespace(dataColumns = dataShape.keys(), modelNamespace = self.modelName).transform(constantTerm)

            #         rhs = Tree("sum", [
            #             Tree("funccall", [
            #                 Token("FUNCNAME", "lag"),
            #                 Tree("arguments", [constantTerm])
            #             ]),
            #             Tree("operator", [
            #                 Token("SUMOPERATOR", "+")
            #             ]),
            #             copy.deepcopy(rhs)
            #         ])
            #     else:
            #         raise ValueError(f"Left-hand side of the assignment must be a variable or a diff. Found: {lhs.data}.")

            # Get data padding
            SetLeafNodeLags().visit(lhs)
            lhsMaxLag = GetMaxLag().transform(lhs)
            SetLeafNodeLags().visit(rhs)
            rhsMaxLag = GetMaxLag().transform(rhs)
            assignment["maxDataLag"] = max(lhsMaxLag, rhsMaxLag)
            assignment["leftPadding"] = assignment["maxDataLag"]
            assignment["rightPadding"] = 0

            # Get parameters names
            parameterShapesGetter = GetParameterShapesAndDimensionLabels(dataShape, 1)
            outputShape = parameterShapesGetter.transform(rhs)
            if len(outputShape) == 1 and outputShape[0] == 1:
                shapes = parameterShapesGetter.parameterShapes
                dimensionLabels = parameterShapesGetter.parameterDimensionLabels

                for name, shape in shapes.items():
                    if name not in self.parameters:
                        self.parameters[name] = {"shape": shape}
                    else:
                        if "shape" not in self.parameters[name]:
                            self.parameters[name]["shape"] = shape
                        elif self.parameters[name]["shape"] != shape:
                            raise ValueError(f"Parameter '{name}' has inconsistent shapes: {self.parameters[name]['shape']} vs {shape}.")
                        
                    if name in dimensionLabels:
                        self.parameters[name]["dimensionLabels"] = dimensionLabels[name]
                    else:
                        self.parameters[name]["dimensionLabels"] = {}
            else:
                raise ValueError(f"Output shape {outputShape[0]} is not compatible with the model.")
            
            # Finalize the formula tokens, deep copy to avoid modifying the original tokens
            lhsReconstructed = ModelFormulaReconstructor.reconstruct(copy.deepcopy(lhs))
            rhsReconstructed = ModelFormulaReconstructor.reconstruct(copy.deepcopy(rhs))
            print("    State assignment:", lhsReconstructed, "=", rhsReconstructed)

            assignment["lhs"] = lhs
            assignment["rhs"] = rhs
          
        # Process optimization functions (should come after formulas)
        for i, optimize in enumerate(self.optimize):
            optimize["type"] = optimize["tokens"].children[0].value
            if optimize["type"] not in ["minimize", "maximize"]:
                raise ValueError(f"Unsupported optimization type '{optimize['type']}'. Supported types are 'minimize' and 'maximize'.")
            
            sum = ExpandFunctions().transform(optimize["tokens"].children[1])
            del optimize["tokens"]
            sum = AddNamespace(dataColumns = dataShape.keys(), modelNamespace = self.modelName).transform(sum)

            SetLeafNodeLags().visit(sum)
            
            # Get parameters names
            parameterShapesGetter = GetParameterShapesAndDimensionLabels(dataShape, 1)
            outputShape = parameterShapesGetter.transform(sum)
            shapes = parameterShapesGetter.parameterShapes
            dimensionLabels = parameterShapesGetter.parameterDimensionLabels
            for name, shape in shapes.items():
                if name not in self.parameters:
                    self.parameters[name] = {"shape": shape}
                else:
                    if "shape" not in self.parameters[name]:
                        self.parameters[name]["shape"] = shape
                    elif self.parameters[name]["shape"] != shape:
                        raise ValueError(f"Parameter '{name}' has inconsistent shapes: {self.parameters[name]['shape']} vs {shape}.")
                    
                if name in dimensionLabels:
                    self.parameters[name]["dimensionLabels"] = dimensionLabels[name]
                else:
                    self.parameters[name]["dimensionLabels"] = {}
            
            # Finalize the formula tokens
            reconstructed = ModelFormulaReconstructor.reconstruct(copy.deepcopy(sum))
            print("    " + optimize['type'] + ":", reconstructed)

            optimize["sum"] = sum

        self.latentParameters = unpackPriors(self.priors, self.parameters, self.parameterizations)

        # Initialize parameters (should come after all formulas and assignments)
        keys = jax.random.split(key, len(self.parameters))
        self.parameterConstants = {}
        self.parameterValues = {}
        for i, name in enumerate(self.parameters):
            if "shape" not in self.parameters[name]:
                self.parameters[name]["shape"] = (1,)  # Default shape if not specified

            shape = self.parameters[name]["shape"]

            if "." in name and name.split(".")[0] != self.modelName and name.split(".")[0] != "guide":
                # Skip parameters that are not part of the model or guide
                continue

            if name in self.latentParameters:
                continue

            if "tokens" in self.parameters[name]:
                valueGetter = GetValueFromData(randomKey = keys[i], params = self.parameterValues)
                value = valueGetter.transform(self.parameters[name]["tokens"])

                if name in self.parameterizations:
                    value = self.parameterizations[name]["inverse"](value)

                if "isConstant" in self.parameters[name] and self.parameters[name]["isConstant"]:
                    self.parameterConstants[name] = jnp.broadcast_to(value, shape)
                else:
                    self.parameterValues[name] = jnp.broadcast_to(value, shape)
            else:
                value = jax.random.normal(keys[i], shape) * 0.1 
                self.parameterValues[name] = jnp.broadcast_to(value, shape)

        for name in self.importedParameters:
            if "." not in name:
                name = self.modelName + "." + name
            if name not in self.parameterValues:
                raise ValueError(f"Imported parameter '{name}' not found in the model parameters. Please ensure the parameter is defined in the formula.")
            value = self.importedParameters[name]
            if name in self.parameterizations:
                value = self.parameterizations[name]["inverse"](value)
            self.parameterValues[name] = value

        for name in self.importedConstants:
            if "." not in name:
                name = self.modelName + "." + name
            value = self.importedConstants[name]
            if name in self.parameterizations:
                value = self.parameterizations[name]["inverse"](value)
            self.parameterConstants[name] = value
                        
        # Check if any parameters are NaN or infinite
        for name, value in self.parameterValues.items():
            if jnp.any(jnp.isnan(value)) or jnp.any(jnp.isinf(value)):
                raise ValueError(f"Parameter '{name}' contains NaN or infinite values. Please check the parameter initialization.")
        for name, value in self.parameterConstants.items():
            if jnp.any(jnp.isnan(value)) or jnp.any(jnp.isinf(value)):
                raise ValueError(f"Constant parameter '{name}' contains NaN or infinite values. Please check the parameter initialization.")
            
        # for parameterization in self.parameterizations:
        #     print(f"Parameterization '{parameterization}' defined with unconstraint names: {self.parameterizations[parameterization]['unconstraintParameterNames']}.")

        # Get global properties
        self.leftPadding = max([assignment["leftPadding"] for assignment in self.assignments]) if len(self.assignments) > 0 else 0
        self.rightPadding = max([assignment["rightPadding"] for assignment in self.assignments]) if len(self.assignments) > 0 else 0
        self.dataShape = dataShape
        # print("leftPadding:", self.leftPadding, "rightPadding:", self.rightPadding)
        # print("dataColumns:", self.dataColumns)
        # print("constants:", self.parameterConstants)
        # print("Starting parameters:")
        # for name, value in self.parameterValues.items():
        #     print(f"  {name}: {value.shape} = {value}")
        # print("Constants:")
        # for name, value in self.parameterConstants.items():
        #     print(f"  {name}: {value.shape} = {value}")

        print()

        super().constructModel()

    def getAssignments(self, modelName = "model"):
        exportedAssignments = []
        for assignment in self.assignments:
            exportedAssignments.append({
                "name": assignment["name"].replace("model.", modelName + "."),
                "lhs": AddNamespace(dataColumns = list(self.dataShape.keys()), modelNamespace = modelName).transform(assignment["lhs"]),
                "rhs": AddNamespace(dataColumns = list(self.dataShape.keys()), modelNamespace = modelName).transform(assignment["rhs"]),
                "leftPadding": assignment["leftPadding"],
                "rightPadding": assignment["rightPadding"],
                "maxDataLag": assignment["maxDataLag"]
            })
        return exportedAssignments

    def addAssignment(self, assignments):
        self.importedAssignments.extend(assignments)

    def getParameters(self, modelName = "model"):
        constants = {}
        for parameter in self.parameterValues:
            constants[parameter.replace("model.", modelName + ".")] = self.parameterValues[parameter]
        return constants
    
    def setParameters(self, parameters, type = "init", modelName = "model"):
        if type == "init":
            for name in parameters:
                if "." not in name:
                    self.importedParameters[modelName + "." + name] = jnp.asarray(parameters[name])
                else:
                    self.importedParameters[name] = jnp.asarray(parameters[name])
        elif type == "const":
            for name in parameters:
                if "." not in name:
                    self.importedConstants[modelName + "." + name] = jnp.asarray(parameters[name])
                else:
                    self.importedConstants[name] = jnp.asarray(parameters[name])
        else:
            raise ValueError(f"Unsupported parameter type '{type}'. Supported types are 'init' and 'const'.")
        
    def getNewParamsFromPrior(self, randomKey):
        newParams = {}
        keys = jax.random.split(randomKey, len(self.latentParameters))
        for i, name in enumerate(self.latentParameters):
            valueGetter = GetValueFromData(randomKey = keys[i], params = newParams, constants = self.parameterConstants, parameterizations = self.parameterizations)
            newParams[name] = valueGetter.transform(self.latentParameters[name]["generate"])

        return newParams

    def requestPadding(self, dataset):
        dataset.setLeftPadding(self.leftPadding)
        dataset.setRightPadding(self.rightPadding)

    @partial(jax.jit, static_argnames=["self", "steps"])
    def predictStep(self, params, randomKey, data = None, steps = None, universalStateInitializer = jnp.nan):
        if data is not None:
            dataLength = data.shape[0]
        else:
            if steps is None:
                raise ValueError("Either 'data' or 'steps' must be provided to predict.")
            dataLength = steps

        if True: #if self.leftPadding > 0:
            states = {
                assignment["name"] : jnp.full((dataLength, 1), universalStateInitializer, dtype=jnp.float32) for assignment in self.importedAssignments
            } | {
                assignment["name"] : jnp.full((dataLength, 1), universalStateInitializer, dtype=jnp.float32) for assignment in self.assignments
            }
            
            for name in self.initializations:
                parameterName = name["name"]
                index = name["index"]
                valueGetter = GetValueFromData(
                    params = params, constants = self.parameterConstants, parameterizations = self.parameterizations
                )
                initialValue = valueGetter.transform(name["value"])
                if parameterName in states:
                    states[parameterName] = states[parameterName].at[index, :].set(initialValue)
                else:
                    raise ValueError(f"Initialization parameter '{parameterName}' not found in assignments. Please ensure the parameter is defined in the formula.")
            
            initialCarry = (states, randomKey)

            def forward(carry, t0):
                states, randomKey = carry

                for assignments in [self.importedAssignments, self.assignments]:
                    for assignment in assignments:
                        valueGetter = GetValueFromData(
                            params = params, constants = self.parameterConstants, 
                            data = data, dataShape = self.dataShape, t0 = t0, states = states,
                            parameterizations = self.parameterizations, randomKey = randomKey)
                        rhs = valueGetter.transform(assignment["rhs"])
                        randomKey = valueGetter.randomKey
                        
                        oldValue = states[assignment["name"]][t0, :]
                        newValue = jnp.where(t0 >= assignment["leftPadding"], rhs, oldValue)
                        states[assignment["name"]] = states[assignment["name"]].at[t0, :].set(newValue)

                carry = states, randomKey

                return carry, {}

            finalCarry, _ = lax.scan(forward, initialCarry, jnp.arange(dataLength))
            states, randomKey = finalCarry

        else:
            # TODO: add random key support for batched forward pass
            raise NotImplementedError("Batched forward pass with random key is not implemented yet.")

            def forward(t0):
                states = {}

                for assignment in self.assignments:
                    valueGetter = GetValueFromData(
                        params = params, constants = self.parameterConstants, data = data, dataShape = self.dataShape, 
                        t0 = t0, statesT0 = states, parameterizations = self.parameterizations)
                    value = valueGetter.transform(assignment["rhs"])
                    states[assignment["name"]] = value

                return states
            
            batchedForward = jax.vmap(forward)
            states = batchedForward(jnp.arange(dataLength))

        return states
    
    @partial(jax.jit, static_argnames=["self", "returnLossPerSample"])
    def lossStep(self, params, data, randomKey = jax.random.PRNGKey(0), returnLossPerSample = False):
        if len(self.latentParameters) == 0:
            return self.MLE_loss(params, data, randomKey, returnLossPerSample)
        else:
            if returnLossPerSample:
                raise ValueError("returnLossPerSample=True is not supported for VI_loss.")
            
            return self.VI_loss(params, data, randomKey)

    @partial(jax.jit, static_argnames=["self", "returnLossPerSample"])
    def MLE_loss(self, params, data, randomKey = jax.random.PRNGKey(0), returnLossPerSample = False):
        if len(self.optimize) == 0:
            raise ValueError("No optimization functions defined. Please ensure the model has at least one optimization function.")

        states = self.predictStep(params, randomKey, data = data, universalStateInitializer = 0.0)

        losses = []

        for optimize in self.optimize:
            def helper(statesT0, t0):
                valueGetter = GetValueFromData(
                    params = params, constants = self.parameterConstants, data = data, dataShape = self.dataShape, 
                    t0 = t0, statesT0 = statesT0, parameterizations = self.parameterizations)
                
                # valueGetter = GetValueFromData(
                #     params = params, constants = self.parameterConstants, data = data, dataShape = self.dataShape, 
                #     t0 = t0, states = states, parameterizations = self.parameterizations, randomKey = randomKey)

                if optimize["type"] == "minimize":
                    return valueGetter.transform(optimize["sum"])
                elif optimize["type"] == "maximize":
                    return -valueGetter.transform(optimize["sum"])
                else:
                    raise ValueError(f"Unsupported optimization type '{optimize['type']}'. Supported types are 'minimize' and 'maximize'.")

            batchedHelper = jax.vmap(helper, in_axes=(0, 0))
            paddedStates = jax.tree_util.tree_map(lambda arr: arr[self.leftPadding:data.shape[0] - self.rightPadding, :], states)
            timesteps = jnp.arange(self.leftPadding, data.shape[0] - self.rightPadding)
            loss = batchedHelper(paddedStates, timesteps)

            lossShape = jnp.shape(loss)
            if len(lossShape) == 1 and lossShape[0] == data.shape[0] - self.leftPadding - self.rightPadding:
                losses.append(loss)
            elif len(lossShape) == 2 and lossShape[0] == data.shape[0] - self.leftPadding - self.rightPadding:
                losses.append(jnp.sum(loss, axis=1))
            else:
                raise ValueError(f"Loss function returned unexpected shape {lossShape}. Expected shape is ({data.shape[0] - self.leftPadding - self.rightPadding},) or ({data.shape[0] - self.leftPadding - self.rightPadding}, n).")

        totalLossPerSample = jax.numpy.sum(jax.numpy.stack(losses, axis=1), axis=1)

        if jnp.shape(totalLossPerSample) != (data.shape[0] - self.leftPadding - self.rightPadding,):
            raise ValueError(f"Total loss per sample has unexpected shape {jnp.shape(totalLossPerSample)}. Expected shape is ({data.shape[0] - self.leftPadding - self.rightPadding},).")
        
        if self.burnInTime > 0:
            totalLossPerSample = totalLossPerSample[self.burnInTime:]

        if returnLossPerSample:
            return totalLossPerSample
        else:
            totalLoss = jax.numpy.mean(totalLossPerSample)
            return totalLoss
    
    @partial(jax.jit, static_argnames=["self"])
    def OLS_getXY(self, data):
        dataLength = data.shape[0]
        X = {}
        Y = {}
        for dependingVariable in self.OLS:            
            x = []
            numIndependentVariables = 0
            for term in self.OLS[dependingVariable]["terms"]:
                def helper(t0):
                    valueGetter = GetValueFromData(constants = self.parameterConstants, data = data, dataShape = self.dataShape, t0 = t0)
                    value = valueGetter.transform(self.OLS[dependingVariable]["terms"][term])
                    return jnp.reshape(value, (-1,))
                batchedHelper = jax.vmap(helper, in_axes=(0,))
                termValue = batchedHelper(jnp.arange(dataLength))
                x.append(termValue)
                numIndependentVariables += np.prod(self.parameterValues[term].shape)
            
            if self.OLS[dependingVariable]["bias"] is not None:
                if self.OLS[dependingVariable]["biasIsPositive"]:
                    bias = jnp.ones((dataLength, 1), dtype=jnp.float32)
                else:
                    bias = -jnp.ones((dataLength, 1), dtype=jnp.float32)
                x.append(bias)
                numIndependentVariables += 1

            X[dependingVariable] = jnp.reshape(jnp.concatenate(x, axis=1), (-1, numIndependentVariables))

            yIndex = list(self.dataShape.keys()).index(dependingVariable)
            Y[dependingVariable] = data[:, yIndex]

            for constant in self.OLS[dependingVariable]["constants"]:
                def helper(t0):
                    valueGetter = GetValueFromData(constants = self.parameterConstants, data = data, dataShape = self.dataShape, t0 = t0)
                    value = valueGetter.transform(constant)
                    return value
                batchedHelper = jax.vmap(helper, in_axes=(0,))
                constantValues = batchedHelper(jnp.arange(dataLength))
                Y[dependingVariable] -= constantValues

            if self.burnInTime > 0:
                X[dependingVariable] = X[dependingVariable][self.burnInTime:]
                Y[dependingVariable] = Y[dependingVariable][self.burnInTime:]

        return X, Y
    
    def VI_sample(self, guideParams, randomKey):
        latentParams = {}

        for name in self.latentParameters:
            subKey, randomKey = jax.random.split(randomKey)
            sampleTokens = self.latentParameters[name]["sample"]
            valueGetter = GetValueFromData(
                params = guideParams, 
                randomKey = subKey, 
                parameterizations = self.parameterizations
            )
            value = valueGetter.transform(sampleTokens)

            if name == "model.re":
                value = value - jnp.mean(value)

            latentParams[name] = value
            
        for name in guideParams:
            if not name.startswith("guide."):
                latentParams[name] = guideParams[name] 

        return latentParams

    def VI_priorProbability(self, latentParams):
        logPriorSum = 0.0

        for name in latentParams:
            if name in self.latentParameters:
                priorTokens = self.latentParameters[name]["prior"]
                valueGetter = GetValueFromData(
                    params = latentParams,
                    parameterizations=self.parameterizations
                )
                logPrior = valueGetter.transform(priorTokens)
                logPriorSum += jnp.sum(logPrior)

        return logPriorSum

    def VI_guideProbability(self, guideParams, latentParams):
        logGuideSum = 0.0

        mergedParams = {}
        for name in self.latentParameters:
            mergedParams[name] = latentParams[name]
        for name in guideParams:
            if name.startswith("guide."):
                mergedParams[name] = guideParams[name]

        for name in self.latentParameters:
            guideTokens = self.latentParameters[name]["guide"]
            valueGetter = GetValueFromData(
                params = mergedParams,
                parameterizations=self.parameterizations
            )
            logGuide = valueGetter.transform(guideTokens)
            logGuideSum += jnp.sum(logGuide)

        return logGuideSum

    @partial(jax.jit, static_argnames=["self"])
    def VI_loss(self, guideParams, data, randomKey):

        # Maximum A Posteriori (MAP) or Empirical Bayes estimation
        #  --> This is the mix of MLE and VI, where we optimize both model parameters and guide parameters

        print("guideParams:", list(guideParams.keys()))

        # 1. Sample latent variables from the guide q(z)
        latentParams = self.VI_sample(guideParams, randomKey)

        print("latentParams:", list(latentParams.keys()))

        # 2. Compute log p(y | z) using the default MLE loss
        logPY = -self.MLE_loss(latentParams, data)

        # 3. Compute log p(z) 
        logPZ = self.VI_priorProbability(latentParams)

        # 4. Compute log p(y, z)
        logPYZ = logPY + logPZ

        # 5. Compute log q(z)
        logQZ = self.VI_guideProbability(guideParams, latentParams)

        # ELBO = E_q[log p(y, z) - log q(z)]
        elbo = logPYZ - logQZ

        # We maximize the ELBO (usually minimize negative ELBO)
        return -jnp.reshape(elbo, ())
    
    @partial(jax.jit, static_argnames=["self"])
    def MCMC_logPosterior(self, params, data):
        logLikelihood = -(self.MLE_loss(params, data) *  data.shape[0])
        logPrior = self.VI_priorProbability(params)
        logPosterior = logLikelihood + logPrior
        return logPosterior
    
    @partial(jax.jit, static_argnames=["self"])
    def MCMC_step(self, initParams, initLogPosterior, data, stepSizePerBlock, key):
        numberOfBlocks = jax.tree_util.tree_leaves(stepSizePerBlock)[0].shape[0]
        componentKeys = jax.random.split(key, numberOfBlocks)

        def componentStep(carry, xsSlice):
            params, logPosterior = carry
            componentKey, stepSizeComponent = xsSlice

            keyProposal, keyAccept = jax.random.split(componentKey, 2)

            randomPytree = makeRandomPytree(keyProposal, params)
            proposedParams = jax.tree_util.tree_map(lambda p, d, s: p + (d * s), params, randomPytree, stepSizeComponent)
            proposedLogPosterior = self.MCMC_logPosterior(proposedParams, data)

            logAlpha = jnp.minimum(0.0, proposedLogPosterior - logPosterior)
            logThreshold = jnp.log(jax.random.uniform(keyAccept))
            accept = logThreshold < logAlpha

            nextParams, nextLogPosterior, acceptCount = jax.lax.cond(accept, 
                lambda: (proposedParams, proposedLogPosterior, 1), 
                lambda: (params, logPosterior, 0)
            )

            carry = (nextParams, nextLogPosterior)
            return carry, acceptCount

        initialCarry = (initParams, initLogPosterior)
        xs = (componentKeys, stepSizePerBlock)
        finalCarry, acceptCounts = lax.scan(componentStep, initialCarry, xs)
        finalParams, finalLogPosterior = finalCarry

        return finalParams, finalLogPosterior, acceptCounts

    @partial(jax.jit, static_argnames=["self", "chunkSize"])
    def MCMC_chunk(self, params, data, stepSizePerBlock, randomKey, chunkSize):
        keys = jax.random.split(randomKey, chunkSize)
        logPosterior = self.MCMC_logPosterior(params, data)

        def scanStep(carry, key):
            params, logPosterior, totalAcceptCount = carry
            nextParams, nextLogPosterior, acceptCount = self.MCMC_step(params, logPosterior, data, stepSizePerBlock, key)
            totalAcceptCount = totalAcceptCount + acceptCount
            carry = (nextParams, nextLogPosterior, totalAcceptCount)

            # Only return the last parameters from each step
            return carry, nextParams

        numberOfBlocks = jax.tree_util.tree_leaves(stepSizePerBlock)[0].shape[0]
        zeroAcceptCount = jnp.array([0] * numberOfBlocks)
        initialCarry = (params, logPosterior, zeroAcceptCount)
        finalCarry, allParams = lax.scan(scanStep, initialCarry, keys)
        lastParams, lastLogPosterior, totalAcceptCount = finalCarry

        return lastParams, allParams, totalAcceptCount
    
    def EM_initialize(self):
        for em in self.EM:
            variable = em["variable"]
            mean = em["mean"]
            std = em["std"]

            likelihoodTokens = Tree(Token("RULE", "funccall"), [
                Token("FUNCNAME", "norm.logpdf"),
                Tree(Token("RULE", "arguments"), [
                    Tree(Token("RULE", "variable"), [
                        Token("VARIABLENAME", variable)
                    ]),
                    mean,
                    std
                ])
            ])

            meanParams = GetParameterNames().transform(mean)
            stdParams = GetParameterNames().transform(std)
            allParams = set(meanParams + stdParams)

            modelParams = []
            latentParams = []

            for param in allParams:
                if "isLatent" in self.parameters[param] and self.parameters[param]["isLatent"]:
                    latentParams.append(param)
                else:
                    modelParams.append(param)

            priors = []

            for param in latentParams:
                if param not in self.latentParameters:
                    raise ValueError(f"Latent parameter '{param}' not found in latent parameters. Please ensure the parameter is defined as latent.")
                priors.append(self.latentParameters[param]["prior"])

            posteriorTokens = Tree(Token("RULE", "sum"), [
                likelihoodTokens,
            ] + priors)

            expr = LarkToSymPy().transform(posteriorTokens)
            print("EM posterior expression for variable", variable, ":", expr)

            expr = expr.expand()

            z = sp.Symbol(latentParams[0])  # Currently only supports one latent parameter
            E_expr = take_expectation(expr, z) 
            print("EM E[posterior] expression for variable", variable, ":", E_expr)

            e_terms = get_expectation_terms(E_expr, z)
            print("E_z terms found:")
            for term in e_terms:
                print("     ", term)





    def EM_step(self):
        pass


