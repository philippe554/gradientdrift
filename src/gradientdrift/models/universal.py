
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

class Universal(Model):
    def __init__(self, formula):
        self.formula = formula
        self.importedAssignments = []
        self.importedConstants = {}
        self.importedParameters = {}

    def constructModel(self, dataColumns, key, modelName = "model"):
        self.dataColumns = dataColumns
        self.modelName = modelName

        # Tokenize the formula
        if type(self.formula) is str:
            formulaTree = getParser().parse(self.formula)
        elif type(self.formula) is Tree:
            formulaTree = self.formula
        else:
            raise ValueError(f"Unsupported formula type: {type(self.formula)}. Expected str or Tree.")
        formulaTree = RemoveNewlines().transform(formulaTree)
        formulaTree = FlattenNamespaces().transform(formulaTree)
        
        statements = formulaTree.children
        parameterDefinition = [{"tokens" : s} for s in statements if s.data == 'parameterdefinition']
        constraintDefinitions = [{"tokens" : s} for s in statements if s.data == 'constraint']
        self.initializations = [{"tokens" : s} for s in statements if s.data == 'initialization']
        self.assignments = [{"tokens" : s} for s in statements if s.data == 'assignment']
        self.formulas = [{"tokens" : s} for s in statements if s.data == 'formula']
        self.optimize = [{"tokens" : s} for s in statements if s.data == 'optimize']

        # Process parameter definitions (should come first)
        self.parameters = {}
        parameterDefinitionsFound = set()
        for i, definition in enumerate(parameterDefinition):
            names = []
            for name in definition["tokens"].children[0].children:
                name = name.children[0].value
                if "." in name:
                    if name.split(".")[0] == modelName:
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
                        print(f"Parameter '{name}' shape set to {shape}.")
                    else: # Set to a constant value. A sum can flatten to multiple token types, thus match anything
                        self.parameters[name]["tokens"] = rhs
                        print(f"Parameter '{name}' initialized to {ModelFormulaReconstructor.reconstruct(copy.deepcopy(rhs))}.")

                    
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
                        parameters = GetParameters().transform(right)
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
            rhs = AddNamespace(dataColumns = [], modelNamespace = self.modelName).transform(rhs)
            initialization["value"] = rhs

            print(f"Initialization '{initialization['name']}' at index {initialization['index']} set to {ModelFormulaReconstructor.reconstruct(copy.deepcopy(initialization['value']))}.")
  
        # Process formulas (insert parameters and split into assignments and optimizations)
        for i, formula in enumerate(self.formulas):
            lhs, rhs = formula["tokens"].children

            if rhs.data == "funccall" and isDistributionFunction(rhs.children[0].value):
                rhsIsProbabilistic = True
            else:
                rhsIsProbabilistic = False

            dependingVariables = GetVariables().transform(lhs)
            for var in dependingVariables:
                if "." in var:
                    raise ValueError(f"LHS of a formula must not contain namespaces.")

            # Name unnamed parameters
            parameters = GetParameters().transform(rhs)
            rhs = NameUnnamedParameters(parameters).transform(rhs)

            # If no parameters found, auto insert them in the top level sum
            if not rhsIsProbabilistic:               
                parameters = GetParameters().transform(rhs)

                if len(parameters) == 0: # No parameters found, automatic mode
                    LabelOuterSum().visit(rhs)
                    rhs = InsertParameters().transform(rhs)
                    parameters = GetParameters().transform(rhs)
                    if len(parameters) == 0:
                        raise ValueError("No parameters found in the formula. Please ensure the formula is correctly specified.")
                    if len(set(parameters)) != len(parameters):
                        raise ValueError("Duplicate parameters found in the formula. Please ensure each parameter is unique.")
            
            
            # Split the formula in an assignment and an optimization function
            if rhsIsProbabilistic:
                distributionName = rhs.children[0].value
                if distributionName == "normal":
                    args = rhs.children[1].children
                    if len(args) != 2:
                        raise ValueError(f"Normal distribution must have exactly two arguments: mean and standard deviation. Found: {len(args)} arguments.")

                    lhs = lhs.children[0] # TODO: Change this when the formula parser supports multiple left-hand sides

                    ### Copy from bellow

                    if lhs.data != "variable": # like a diff(y) ~ normal(...)
                        lhsParameters = GetParameters().transform(lhs)
                        if len(lhsParameters) != 0:
                            raise ValueError(f"Left-hand side of the assignment must not contain parameters. Found: {', '.join(lhsParameters)}.")
                        lhsVariables = GetVariables().transform(lhs)
                        if len(set(lhsVariables)) != 1:
                            raise ValueError(f"Left-hand side of the assignment must contain exactly one variable. Found: {', '.join(lhsVariables)}.")

                        if lhs.data == "funccall" and lhs.children[0].value == "diff":
                            lhsInterated = lhs.children[1].children[0]

                            # lhs will be on the rhs, so remove the namespace and add it again based on data columns available
                            #constantTerm = copy.deepcopy(lhs)
                            #constantTerm.children[0].value = constantTerm.children[0].value.replace("model.", "") # TODO
                            constantTerm = Tree(Token('RULE', 'variable'), [Token('NAME', lhsInterated.children[0].value)])
                            constantTerm = AddNamespace(dataColumns = self.dataColumns, modelNamespace = self.modelName).transform(constantTerm)

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
                        else:
                            raise ValueError(f"Left-hand side of the assignment must be a variable or a diff. Found: {lhs.data}.")
                    else:
                        lhsInterated = copy.deepcopy(lhs)
                        rhsAssignment = copy.deepcopy(rhs)

                    ###



                    tokens = Tree(Token("RULE", "assignment"), [lhsInterated, rhsAssignment])
                    self.assignments.append({
                        "tokens": tokens
                    })
                    
                    # meanReconstructed = ModelFormulaReconstructor.reconstruct(copy.deepcopy(args[0]))
                    # stdReconstructed = ModelFormulaReconstructor.reconstruct(copy.deepcopy(args[1]))
                    # tokens = getParser("optimize").parse(
                    #     "maximize: norm.logpdf(" + lhsReconstructed + ", " + meanReconstructed + ", " + stdReconstructed + ")"
                    # )
                    # tokens = FlattenNamespaces().transform(tokens)

                    # Untested code: TODO

                    lhsData = AddNamespace(dataColumns = self.dataColumns, modelNamespace = self.modelName).transform(lhs)
                    args = rhs.children[1].children
                    rhsMean = args[0]
                    rhsStd = args[1]

                    tokens = Tree(Token("RULE", "optimize"), [
                        Token("OPTIMIZETYPE", "maximize"),
                        Tree(Token("RULE", "funccall"), [
                            Token("FUNCNAME", "norm.logpdf"),
                            Tree(Token("RULE", "arguments"), [
                                lhsData,
                                rhsMean,
                                rhsStd
                            ])
                        ])
                    ])
                    self.optimize.append({"tokens": tokens})
                else:
                    raise ValueError(f"Unsupported distribution '{distributionName}' in the formula. Supported distributions are: normal, uniform, poisson, binomial, exponential, gamma, beta, cauchy, laplace, lognormal.")
            else:
                lhs = lhs.children[0] # TODO: Change this when the formula parser supports multiple left-hand sides

                ### Copy from bellow

                if lhs.data != "variable":
                    lhsParameters = GetParameters().transform(lhs)
                    if len(lhsParameters) != 0:
                        raise ValueError(f"Left-hand side of the assignment must not contain parameters. Found: {', '.join(lhsParameters)}.")
                    lhsVariables = GetVariables().transform(lhs)
                    if len(set(lhsVariables)) != 1:
                        raise ValueError(f"Left-hand side of the assignment must contain exactly one variable. Found: {', '.join(lhsVariables)}.")

                    if lhs.data == "funccall" and lhs.children[0].value == "diff":
                        lhs = lhs.children[1].children[0]

                        # lhs will be on the rhs, so remove the namespace and add it again based on data columns available
                        #constantTerm = copy.deepcopy(lhs)
                        #constantTerm.children[0].value = constantTerm.children[0].value.replace("model.", "") # TODO
                        constantTerm = Tree(Token('RULE', 'variable'), [Token('NAME', lhs.children[0].value)])
                        constantTerm = AddNamespace(dataColumns = self.dataColumns, modelNamespace = self.modelName).transform(constantTerm)

                        rhs = Tree(Token("RULE", "sum"), [
                            Tree(Token("RULE", "funccall"), [
                                Token("FUNCNAME", "lag"),
                                Tree(Token("RULE", "arguments"), [constantTerm])
                            ]),
                            Tree("operator", [
                                Token("SUMOPERATOR", "+")
                            ]),
                            copy.deepcopy(rhs)
                        ])
                    else:
                        raise ValueError(f"Left-hand side of the assignment must be a variable or a diff. Found: {lhs.data}.")


                ###

                dependingVariables = GetVariables().transform(lhs)
                if len(dependingVariables) != 1:
                    raise ValueError(f"Left-hand side of the formula must contain exactly one variable. Found: {', '.join(dependingVariables)}.")
                dependingVariable = dependingVariables[0]

                lhsModel = AddNamespace(dataColumns = [], modelNamespace = self.modelName).transform(lhs)
                lhsData = AddNamespace(dataColumns = self.dataColumns, modelNamespace = self.modelName).transform(lhs)
                rhs = AddNamespace(dataColumns = self.dataColumns, modelNamespace = self.modelName).transform(rhs)

                tokens = Tree(Token("RULE", "assignment"), [lhsModel, rhs])
                self.assignments.append({"tokens": tokens})

                tokens = Tree(Token("RULE", "optimize"), [
                    Token("OPTIMIZETYPE", "maximize"),
                    Tree(Token("RULE", "funccall"), [
                        Token("FUNCNAME", "norm.logpdf"),
                        Tree(Token("RULE", "arguments"), [
                            lhsModel,
                            lhsData,
                            Tree(Token("RULE", "parameter"), [
                               Token("NAME", "model.sigma_" + dependingVariable)
                            ])
                        ])
                    ])
                ])
                self.optimize.append({"tokens": tokens})

                self.parameterizations["model.sigma_" + dependingVariable] = {
                    "unconstraintParameterNames": ["model.sigma_" + dependingVariable],
                    "apply": lambda p, v = 0: jax.nn.softplus(p) + v,
                    "inverse": lambda p, v = 0: jnp.log(jnp.exp(p - v) - 1)
                }

        for importedAssignment in self.importedAssignments:
            print(f"Imported assignment: {importedAssignment['name']} = {ModelFormulaReconstructor.reconstruct(copy.deepcopy(importedAssignment['rhs']))}")

        # Process assignments (should come after formulas)
        for i, assignment in enumerate(self.assignments):
            lhs, rhs = assignment["tokens"].children
            del assignment["tokens"]

            lhs = AddNamespace(dataColumns = [], modelNamespace = self.modelName).transform(lhs)
            rhs = AddNamespace(dataColumns = self.dataColumns, modelNamespace = self.modelName).transform(rhs)

            rhs = ExpandFunctions().transform(rhs)

            if lhs.data == "variable":
                assignment["name"] = lhs.children[0].value
            else:
                lhsParameters = GetParameters().transform(lhs)
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
            #         constantTerm = AddNamespace(dataColumns = self.dataColumns, modelNamespace = self.modelName).transform(constantTerm)

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
            SetLeafNodeLags().visit(rhs)
            assignment["maxDataLag"] = max(GetMaxLag().transform(rhs), GetMaxLag().transform(lhs))
            assignment["leftPadding"] = assignment["maxDataLag"]
            assignment["rightPadding"] = 0

            # Get parameters names
            parameters = GetParameters().transform(rhs)
            parameterShapesGetter = GetParameterShapes(1)
            outputShape = parameterShapesGetter.transform(rhs)
            if len(outputShape) == 1 and outputShape[0] == 1:
                shapes = parameterShapesGetter.parameterShapes
                for name, shape in shapes.items():
                    if name not in self.parameters:
                        self.parameters[name] = {"shape": shape}
                    else:
                        if "shape" not in self.parameters[name]:
                            self.parameters[name]["shape"] = shape
                        elif self.parameters[name]["shape"] != shape:
                            raise ValueError(f"Parameter '{name}' has inconsistent shapes: {self.parameters[name]['shape']} vs {shape}.")
            else:
                raise ValueError(f"Output shape {outputShape[0]} is not compatible with the model.")
            
            # Finalize the formula tokens, deep copy to avoid modifying the original tokens
            lhsReconstructed = ModelFormulaReconstructor.reconstruct(copy.deepcopy(lhs))
            rhsReconstructed = ModelFormulaReconstructor.reconstruct(copy.deepcopy(rhs))
            print(lhsReconstructed, "=", rhsReconstructed)

            assignment["lhs"] = lhs
            assignment["rhs"] = rhs
          
        # Process optimization functions (should come after formulas)
        for i, optimize in enumerate(self.optimize):
            optimize["type"] = optimize["tokens"].children[0].value
            if optimize["type"] not in ["minimize", "maximize"]:
                raise ValueError(f"Unsupported optimization type '{optimize['type']}'. Supported types are 'minimize' and 'maximize'.")
            
            sum = ExpandFunctions().transform(optimize["tokens"].children[1])
            del optimize["tokens"]
            sum = FlattenNamespaces().transform(sum)
            sum = AddNamespace(dataColumns = self.dataColumns, modelNamespace = self.modelName).transform(sum)

            SetLeafNodeLags().visit(sum)
            
            # Get parameters names
            parameterShapesGetter = GetParameterShapes(1)
            outputShape = parameterShapesGetter.transform(sum)
            shapes = parameterShapesGetter.parameterShapes
            for name, shape in shapes.items():
                if name not in self.parameters:
                    self.parameters[name] = {"shape": shape}
                else:
                    if "shape" not in self.parameters[name]:
                        self.parameters[name]["shape"] = shape
                    elif self.parameters[name]["shape"] != shape:
                        raise ValueError(f"Parameter '{name}' has inconsistent shapes: {self.parameters[name]['shape']} vs {shape}.")
                    
            
            # Finalize the formula tokens
            reconstructed = ModelFormulaReconstructor.reconstruct(copy.deepcopy(sum))
            print(optimize['type'] + ":", reconstructed)

            optimize["sum"] = sum

        # Initialize parameters (should come after all formulas and assignments)
        keys = jax.random.split(key, len(self.parameters))
        self.parameterConstants = {}
        self.parameterValues = {}
        for i, name in enumerate(self.parameters):
            if "shape" not in self.parameters[name]:
                self.parameters[name]["shape"] = (1,)  # Default shape if not specified

            shape = self.parameters[name]["shape"]

            if "." in name and name.split(".")[0] != self.modelName:
                print(f"Skipping parameter '{name}' as it is imported from another model.")
                continue

            if "tokens" in self.parameters[name]:
                valueGetter = GetValueFromData(randomKey = keys[i])
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
            
        for parameterization in self.parameterizations:
            print(f"Parameterization '{parameterization}' defined with unconstraint names: {self.parameterizations[parameterization]['unconstraintParameterNames']}.")

        # Get global properties
        self.leftPadding = max([assignment["leftPadding"] for assignment in self.assignments])
        self.rightPadding = max([assignment["rightPadding"] for assignment in self.assignments])
        print("leftPadding:", self.leftPadding, "rightPadding:", self.rightPadding)
        print("dataColumns:", self.dataColumns)
        print("constants:", self.parameterConstants)
        print("Starting parameters:")
        for name, value in self.parameterValues.items():
            print(f"  {name}: {value.shape} = {value}")
        print("Constants:")
        for name, value in self.parameterConstants.items():
            print(f"  {name}: {value.shape} = {value}")

        super().constructModel()

    def getAssignments(self, modelName = "model"):
        exportedAssignments = []
        for assignment in self.assignments:
            exportedAssignments.append({
                "name": assignment["name"].replace("model.", modelName + "."),
                "lhs": AddNamespace(dataColumns = self.dataColumns, modelNamespace = modelName).transform(assignment["lhs"]),
                "rhs": AddNamespace(dataColumns = self.dataColumns, modelNamespace = modelName).transform(assignment["rhs"]),
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
                            data = data, dataColumns = self.dataColumns, t0 = t0, states = states,
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
                        params = params, constants = self.parameterConstants, data = data, dataColumns = self.dataColumns, 
                        t0 = t0, statesT0 = states, parameterizations = self.parameterizations)
                    value = valueGetter.transform(assignment["rhs"])
                    states[assignment["name"]] = value

                return states
            
            batchedForward = jax.vmap(forward)
            states = batchedForward(jnp.arange(dataLength))

        return states

    @partial(jax.jit, static_argnames=["self", "returnLossPerSample"])
    def lossStep(self, params, data, randomKey = jax.random.PRNGKey(0), returnLossPerSample = False):
        states = self.predictStep(params, randomKey, data = data, universalStateInitializer = 0.0)

        # jax.debug.print("===================================")
        # jax.debug.print("Parameters: {params}", params=params)
        # jax.debug.print("model.y contains NaN: {contains_nan}", contains_nan=jnp.any(jnp.isnan(states["model.y"])))
        # jax.debug.print("model.sigmaSq contains NaN: {contains_nan}", contains_nan=jnp.any(jnp.isnan(states["model.sigmaSq"])))
        # jax.debug.print("States after prediction: {states}", states=states)

        losses = []

        for optimize in self.optimize:
            def helper(statesT0, t0):
                valueGetter = GetValueFromData(
                    params = params, constants = self.parameterConstants, data = data, dataColumns = self.dataColumns, 
                    t0 = t0, statesT0 = statesT0, parameterizations = self.parameterizations)
                
                # valueGetter = GetValueFromData(
                #     params = params, constants = self.parameterConstants, data = data, dataColumns = self.dataColumns, 
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

            # jax.debug.print("Loss contains NaN: {contains_nan}", contains_nan=jnp.any(jnp.isnan(loss)))
            # jax.debug.print("Loss = {loss}", loss=loss)

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
        
        burnInTime = 100
        totalLossPerSample = totalLossPerSample[burnInTime:]

        # jax.debug.print("Total loss per sample contains NaN: {contains_nan}", contains_nan=jnp.any(jnp.isnan(totalLossPerSample)))
        # jax.debug.print("Total loss per sample = {total_loss_per_sample}", total_loss_per_sample=totalLossPerSample)

        if returnLossPerSample:
            return totalLossPerSample
        else:
            totalLoss = jax.numpy.mean(totalLossPerSample)
            return totalLoss