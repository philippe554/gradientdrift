
import copy
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

from gradientdrift.models.model import Model
from gradientdrift.utils.formulawalkers import *
from gradientdrift.utils.formulaparsers import ModelFormulaParser, ModelFormulaReconstructor

class Universal(Model):
    def __init__(self, formula):
        self.constructModel(formula)

    def constructModel(self, formula, key = jax.random.PRNGKey(42)):
        # Tokenize the formula
        formulaTree = ModelFormulaParser.parse(formula)
        formulaTree = RemoveNewlines().transform(formulaTree)
        
        statements = formulaTree.children
        parameterDefinition = [{"tokens" : s} for s in statements if s.data == 'parameterdefinition']
        constraintDefinitions = [{"tokens" : s} for s in statements if s.data == 'constraint']
        self.initializations = [{"tokens" : s} for s in statements if s.data == 'initialization']
        self.assignments = [{"tokens" : s} for s in statements if s.data == 'assignment']
        self.formulas = [{"tokens" : s} for s in statements if s.data == 'formula']
        self.optimize = [{"tokens" : s} for s in statements if s.data == 'optimize']

        # Process parameter definitions
        self.parameters = {}
        parameterDefinitionsFound = set()
        for i, definition in enumerate(parameterDefinition):
            names = [e.children[0].value for e in definition["tokens"].children[0].children]
            if any(name in parameterDefinitionsFound for name in names):
                raise ValueError(f"Duplicate parameter definition found for names: {', '.join(names)}. Each parameter must be defined only once.")
            parameterDefinitionsFound.update(names)
            for name in names:
                self.parameters[name] = {}
            for rhs in definition["tokens"].children[1:]:
                if rhs.data == "shape":
                    shape = tuple([int(e.value) for e in rhs.children])
                    for name in names:
                        self.parameters[name]["shape"] = shape
                else: # a sum can flatten to multiple token types, thus match anything
                    for name in names:
                        self.parameters[name]["tokens"] = rhs
                    
        # Process constraints
        self.parameterizations = {}
        for i, constraint in enumerate(constraintDefinitions):
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
            initialization["index"] = int(initialization["tokens"].children[1].value)
            initialization["value"] = initialization["tokens"].children[2]

        # Process assignments
        for i, assignment in enumerate(self.assignments):
            lhs, rhs = assignment["tokens"].children

            # Get depending variables and data columns
            assignment["name"] = lhs.value 

            # Get data padding
            SetLeafNodeLags().visit(rhs)
            assignment["maxDataLag"] = GetMaxLag().transform(rhs)
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
            
            # Finalize the formula tokens
            rhsReconstructed = ModelFormulaReconstructor.reconstruct(copy.deepcopy(rhs))
            print(assignment["name"], "=", rhsReconstructed)
            
        # Process formulas
        for i, formula in enumerate(self.formulas):
            lhs, rhs = formula["tokens"].children

            # Get depending variables and data columns
            formula["dependingVariables"] = [e.children[0].value for e in lhs.children]

            # Get data padding
            SetLeafNodeLags().visit(rhs)
            formula["maxDataLag"] = GetMaxLag().transform(rhs)
            formula["maxMALag"] = 0
            formula["leftPadding"] = max(formula["maxDataLag"], formula["maxMALag"])
            formula["rightPadding"] = 0

            # Get parameters names
            parameters = GetParameters().transform(rhs)
            rhs = NameUnnamedParameters(parameters).transform(rhs)
            parameters = GetParameters().transform(rhs)

            if len(parameters) == 0: # No parameters found, automatic mode
                LabelOuterSum().visit(rhs)
                rhs = InsertParameters().transform(rhs)
                parameters = GetParameters().transform(rhs)
                if len(parameters) == 0:
                    raise ValueError("No parameters found in the formula. Please ensure the formula is correctly specified.")
                if len(set(parameters)) != len(parameters):
                    raise ValueError("Duplicate parameters found in the formula. Please ensure each parameter is unique.")
                parameters = set(parameters)

            # Get parameters shapes
            parameterShapesGetter = GetParameterShapes(len(formula["dependingVariables"]))
            outputShape = parameterShapesGetter.transform(rhs)
            if len(outputShape) == 1 and outputShape[0] == len(formula["dependingVariables"]):
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
            
            # Insert back into the object
            formula["tokens"].children = [lhs, rhs]

            # Finalize the formula tokens
            lhsReconstructed = ModelFormulaReconstructor.reconstruct(copy.deepcopy(lhs))
            rhsReconstructed = ModelFormulaReconstructor.reconstruct(copy.deepcopy(rhs))
            print(lhsReconstructed, "~", rhsReconstructed)

        for i, optimize in enumerate(self.optimize):
            optimize["type"] = optimize["tokens"].children[0].value
            if optimize["type"] not in ["minimize", "maximize"]:
                raise ValueError(f"Unsupported optimization type '{optimize['type']}'. Supported types are 'minimize' and 'maximize'.")
            sum = optimize["tokens"].children[1]

            SetLeafNodeLags().visit(sum)
            maxLag = GetMaxLag().transform(sum)
            if maxLag != 0:
                raise ValueError(f"Optimization function cannot have a lag. Found max lag: {maxLag}. Please ensure the optimization function is defined without lags.")
            
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

        keys = jax.random.split(key, len(self.parameters))
        self.parameterConstants = {}
        self.parameterValues = {}
        for i, name in enumerate(self.parameters):
            if "shape" not in self.parameters[name]:
                raise ValueError(f"Parameter '{name}' does not have a shape defined. Please ensure the parameter is defined with a shape or can be derived from the formula.")
            shape = self.parameters[name]["shape"]

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
                
        # Check if any parameters are NaN or infinite
        for name, value in self.parameterValues.items():
            if jnp.any(jnp.isnan(value)) or jnp.any(jnp.isinf(value)):
                raise ValueError(f"Parameter '{name}' contains NaN or infinite values. Please check the parameter initialization.")
        for name, value in self.parameterConstants.items():
            if jnp.any(jnp.isnan(value)) or jnp.any(jnp.isinf(value)):
                raise ValueError(f"Constant parameter '{name}' contains NaN or infinite values. Please check the parameter initialization.")

        # Get global properties
        self.leftPadding = max([formula["leftPadding"] for formula in self.formulas] + 
                               [assignment["leftPadding"] for assignment in self.assignments])
        self.rightPadding = max([formula["rightPadding"] for formula in self.formulas] + 
                                [assignment["rightPadding"] for assignment in self.assignments])
        
        # Initialize data columns to None, in case predict is called as a standalone method
        self.dataColumns = None

        super().constructModel()

    def requestPadding(self, dataset):
        dataset.setLeftPadding(self.leftPadding)
        dataset.setRightPadding(self.rightPadding)
        self.dataColumns = dataset.columns # Move this to a better place if needed

    @partial(jax.jit, static_argnames=["self", "steps"])
    def predict(self, params = None, data = None, steps = None):
        if params is None:
            params = self.parameterValues
        
        if data is not None:
            dataLength = data.shape[0]
        else:
            if steps is None:
                raise ValueError("Either 'data' or 'steps' must be provided to predict.")
            dataLength = steps

        # TODO: better random key handling
        randomKey = jax.random.PRNGKey(0) 

        if self.leftPadding > 0:
            states = {assignment["name"] : jnp.zeros((dataLength, 1)) for assignment in self.assignments}
            
            for name in self.initializations:
                parameterName = name["name"]
                index = name["index"]
                valueGetter = GetValueFromData(
                    params = params, constants = self.parameterConstants, parameterizations = self.parameterizations)
                initialValue = valueGetter.transform(name["value"])
                if parameterName in states:
                    states[parameterName] = states[parameterName].at[index, :].set(initialValue)
                else:
                    raise ValueError(f"Initialization parameter '{parameterName}' not found in assignments. Please ensure the parameter is defined in the formula.")
            
            responses = {i: jnp.zeros((dataLength, len(formula["dependingVariables"]))) for i, formula in enumerate(self.formulas)}
            initialCarry = (states, responses, randomKey)

            def forward(carry, t0):
                states, responses, randomKey = carry

                for i, assignment in enumerate(self.assignments):
                    oldValue = states[assignment["name"]][t0, :]

                    valueGetter = GetValueFromData(
                        params = params, constants = self.parameterConstants, 
                        data = data, dataColumns = self.dataColumns, t0 = t0, states = states,
                        parameterizations = self.parameterizations, randomKey = randomKey)
                    newValue = valueGetter.transform(assignment["tokens"].children[1])
                    randomKey = valueGetter.randomKey

                    value = jnp.where(t0 >= assignment["leftPadding"], newValue, oldValue)
                    states[assignment["name"]] = states[assignment["name"]].at[t0, :].set(value)

                for i, formula in enumerate(self.formulas):
                    oldValue = responses[i][t0, :]

                    valueGetter = GetValueFromData(
                        params = params, constants = self.parameterConstants, data = data, dataColumns = self.dataColumns, 
                        t0 = t0, states = states, parameterizations = self.parameterizations,
                        randomKey = randomKey)
                    newValue = valueGetter.transform(formula["tokens"].children[1])
                    randomKey = valueGetter.randomKey

                    value = jnp.where(t0 >= formula["leftPadding"], newValue, oldValue)
                    responses[i] = responses[i].at[t0, :].set(value)

                carry = states, responses, randomKey

                return carry, {}

            finalCarry, _ = lax.scan(forward, initialCarry, jnp.arange(dataLength))
            states, responses, randomKey = finalCarry

        else:
            # TODO: add random key support for batched forward pass

            def forward(t0):
                states = {}
                responses = {}

                for i, assignment in enumerate(self.assignments):
                    valueGetter = GetValueFromData(
                        params = params, constants = self.parameterConstants, data = data, dataColumns = self.dataColumns, 
                        t0 = t0, statesT0 = states, parameterizations = self.parameterizations)
                    value = valueGetter.transform(assignment["tokens"].children[1])
                    states[assignment["name"]] = value

                for i, formula in enumerate(self.formulas):
                    valueGetter = GetValueFromData(
                        params = params, constants = self.parameterConstants, data = data, dataColumns = self.dataColumns, 
                        t0 = t0, statesT0 = states, parameterizations = self.parameterizations)
                    value = valueGetter.transform(formula["tokens"].children[1])
                    responses[i] = value

                return states, responses
            
            batchedForward = jax.vmap(forward)
            states, responses = batchedForward(jnp.arange(dataLength))

        return states, responses

    @partial(jax.jit, static_argnames=["self", "returnLossPerSample"])
    def loss(self, params, data, returnLossPerSample = False):
        states, responses = self.predict(params, data)

        # for stateName, state in states.items():
        #     print(f"State '{stateName}' shape: {jnp.shape(state)}")

        # for i, response in responses.items():
        #     print(f"Response {i} shape: {jnp.shape(response)}")

        losses = []

        for optimize in self.optimize:
            def helper(statesT0, t0):
                valueGetter = GetValueFromData(
                    params = params, constants = self.parameterConstants, data = data, dataColumns = self.dataColumns, 
                    t0 = t0, statesT0 = statesT0, parameterizations = self.parameterizations)

                if optimize["type"] == "minimize":
                    return valueGetter.transform(optimize["tokens"].children[1])
                elif optimize["type"] == "maximize":
                    return -valueGetter.transform(optimize["tokens"].children[1])
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
        
        burnInTime = 100
        totalLossPerSample = totalLossPerSample[burnInTime:]

        if returnLossPerSample:
            return totalLossPerSample
        else:
            totalLoss = jax.numpy.mean(totalLossPerSample)
            return totalLoss