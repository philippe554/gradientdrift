
import copy
import jax
import jax.numpy as jnp
from jax import lax

from gradientdrift.models.model import Model
from gradientdrift.utils.formulawalkers import *
from gradientdrift.utils.formulaparsers import ModelFormulaParser, ModelFormulaReconstructor

class Universal(Model):
    def __init__(self, formula):
        self.constructModel(formula)

    def constructModel(self, formula):
        # Tokenize the formula
        formulaTree = ModelFormulaParser.parse(formula)
        formulaTree = RemoveNewlines().transform(formulaTree)
        
        statements = formulaTree.children
        parameterDefinition = [{"tokens" : s} for s in statements if s.data == 'parameterdefinition']
        constraintDefinitions = [{"tokens" : s} for s in statements if s.data == 'constraint']
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
                self.parameters[name]["tokens"] = definition["tokens"]
            for i in range(1, len(definition["tokens"].children), 2):
                operator = definition["tokens"].children[i].value
                if operator == "=":
                    rhs = definition["tokens"].children[i + 1]
                    if rhs.data == "shape":
                        shape = tuple([int(e.value) for e in rhs.children])
                        for name in names:
                            self.parameters[name]["shape"] = shape
                    else: # a sum can flatten to multiple token types, thus match anything
                        valueGetter = GetValueFromData()
                        constant = valueGetter.transform(rhs)
                        for name in names:
                            self.parameters[name]["constant"] = constant
                elif operator == "~":
                    initializer = definition["tokens"].children[i + 1]
                    for name in names:
                        self.parameters[name]["initializer"] = initializer
                else:
                    raise ValueError(f"Unsupported parameter definition operator '{operator}'. Supported operators are '=' and '~'.")
                    
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
                                "apply": lambda p: jax.nn.softplus(p) + value
                            }
                    else:
                        raise ValueError(f"Right-hand side of bound constraint must be a parameter list. Found: {right.data}.")
                else:
                    raise ValueError(f"Bound constraint must have exactly 3 children: left-hand side, operator, and right-hand side. Found {len(constraintTokens)} children.")
            else:
                raise ValueError(f"Unsupported constraint type '{constraint['tokens'].data}'. Supported types are 'boundconstraint' and 'sumconstraint'.")
            
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

        if len(self.optimize) == 0:
            raise ValueError("No optimization function found in the formula. Please ensure the formula contains an optimization function.")

        self.parameterConstants = {}
        self.parameterValues = {}
        for name in self.parameters:
            if "shape" not in self.parameters[name]:
                raise ValueError(f"Parameter '{name}' does not have a shape defined. Please ensure the parameter is defined with a shape or can be derived from the formula.")
            shape = self.parameters[name]["shape"]

            if "constant" in self.parameters[name] and "initializer" in self.parameters[name]:
                raise ValueError(f"Parameter '{name}' has both 'constant' and 'initializer' defined. Please define only one of them.")

            if "constant" in self.parameters[name]:
                valueGetter = GetValueFromData()
                constant = valueGetter.transform(self.parameters[name]["constant"])
                self.parameterConstants[name] = constant
            else:
                self.parameterValues[name] = jnp.ones(shape)

        # Get global properties
        self.leftPadding = max([formula["leftPadding"] for formula in self.formulas] + 
                               [assignment["leftPadding"] for assignment in self.assignments])
        self.rightPadding = max([formula["rightPadding"] for formula in self.formulas] + 
                                [assignment["rightPadding"] for assignment in self.assignments])
        
        super().constructModel()

    def requestPadding(self, dataset):
        dataset.setLeftPadding(self.leftPadding)
        dataset.setRightPadding(self.rightPadding)
        self.dataColumns = dataset.columns # Move this to a better place if needed

    def setRandomParameters(self, key):
        numberOfParameters = len(self.parameterValues)
        keys = jax.random.split(key, numberOfParameters)
        
        for i, name in enumerate(self.parameterValues):
            if "initializer" in self.parameters[name]:
                initializer = self.parameters[name]["initializer"]
                valueGetter = GetValueFromData()
                initialValue = jnp.asarray(valueGetter.transform(initializer))
                initializerShape = jnp.shape(initialValue)
                if initializerShape != jnp.shape(self.parameters[name]["shape"]):
                    self.parameterValues[name] = jnp.broadcast_to(initialValue, self.parameters[name]["shape"])
                else:
                    self.parameterValues[name] = initialValue
            else:
                self.parameterValues[name] = jax.random.normal(keys[i], jnp.shape(self.parameters[name]["shape"])) * 0.1

    def predict(self, params, data):
        dataLength = data.shape[0]

        if self.leftPadding > 0:
            states = {assignment["name"] : jnp.zeros((dataLength, 1)) for assignment in self.assignments}
            responses = {i: jnp.zeros((dataLength, len(formula["dependingVariables"]))) for i, formula in enumerate(self.formulas)}
            initialCarry = (states, responses)

            def forward(carry, t0):
                states, responses = carry

                for i, assignment in enumerate(self.assignments):
                    oldValue = states[assignment["name"]][t0, :]

                    valueGetter = GetValueFromData(
                        params = params, constants = self.parameterConstants, 
                        data = data, dataColumns = self.dataColumns, t0 = t0, states = states,
                        parameterizations = self.parameterizations)
                    newValue = valueGetter.transform(assignment["tokens"].children[1])

                    value = jnp.where(t0 >= assignment["leftPadding"], newValue, oldValue)
                    states[assignment["name"]] = states[assignment["name"]].at[t0, :].set(value)

                for i, formula in enumerate(self.formulas):
                    oldValue = responses[i][t0, :]

                    valueGetter = GetValueFromData(
                        params = params, constants = self.parameterConstants, data = data, dataColumns = self.dataColumns, 
                        t0 = t0, states = states, parameterizations = self.parameterizations)
                    newValue = valueGetter.transform(formula["tokens"].children[1])

                    value = jnp.where(t0 >= formula["leftPadding"], newValue, oldValue)
                    responses[i] = responses[i].at[t0, :].set(value)

                carry = states, responses

                return carry, {}

            finalCarry, _ = lax.scan(forward, initialCarry, jnp.arange(dataLength))
            states, responses = finalCarry

        else:
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
        
        if returnLossPerSample:
            return totalLossPerSample
        else:
            totalLoss = jax.numpy.mean(totalLossPerSample)
            return totalLoss