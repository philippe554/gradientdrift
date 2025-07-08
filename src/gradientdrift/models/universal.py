
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
        # Tokenize the formula
        formulaTree = ModelFormulaParser.parse(formula)
        formulaTree = RemoveNewlines().transform(formulaTree)
        #print("Parsed formula tokens:\n" + formulaTree.pretty())
        
        statements = formulaTree.children
        self.assignments = [{"tokens" : s} for s in statements if s.data == 'assignment']
        self.formulas = [{"tokens" : s} for s in statements if s.data == 'formula']
        self.optimize = [{"tokens" : s} for s in statements if s.data == 'optimize']

        # Initialize parameters
        self.params = {}
        self.paramShapes = {}

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

            # Get coefficients names
            coefficients = GetCoefficients().transform(rhs)
            coefficientShapesGetter = GetCoefficientShapes(1)
            outputShape = coefficientShapesGetter.transform(rhs)
            if len(outputShape) == 1 and outputShape[0] == 1:
                shapes = coefficientShapesGetter.coefficientShapes
                for name, shape in shapes.items():
                    if name not in self.paramShapes:
                        self.paramShapes[name] = shape
                    else:
                        if self.paramShapes[name] != shape:
                            raise ValueError(f"Coefficient '{name}' has inconsistent shapes: {self.paramShapes[name]} vs {shape}.")
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

            # Get coefficients names
            coefficients = GetCoefficients().transform(rhs)
            rhs = NameUnnamedCoefficients(coefficients).transform(rhs)
            coefficients = GetCoefficients().transform(rhs)

            if len(coefficients) == 0: # No coefficients found, automatic mode
                LabelOuterSum().visit(rhs)
                rhs = InsertCoefficients().transform(rhs)
                coefficients = GetCoefficients().transform(rhs)
                if len(coefficients) == 0:
                    raise ValueError("No coefficients found in the formula. Please ensure the formula is correctly specified.")
                if len(set(coefficients)) != len(coefficients):
                    raise ValueError("Duplicate coefficients found in the formula. Please ensure each coefficient is unique.")
                coefficients = set(coefficients)

            # Get coefficients shapes
            coefficientShapesGetter = GetCoefficientShapes(len(formula["dependingVariables"]))
            outputShape = coefficientShapesGetter.transform(rhs)
            if len(outputShape) == 1 and outputShape[0] == len(formula["dependingVariables"]):
                shapes = coefficientShapesGetter.coefficientShapes
                for name, shape in shapes.items():
                    if name not in self.paramShapes:
                        self.paramShapes[name] = shape
                    else:
                        if self.paramShapes[name] != shape:
                            raise ValueError(f"Coefficient '{name}' has inconsistent shapes: {self.paramShapes[name]} vs {shape}.")
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
            
            # Get coefficients names
            coefficientShapesGetter = GetCoefficientShapes(1)
            outputShape = coefficientShapesGetter.transform(sum)
            shapes = coefficientShapesGetter.coefficientShapes
            for name, shape in shapes.items():
                if name not in self.paramShapes:
                    self.paramShapes[name] = shape
                else:
                    if self.paramShapes[name] != shape:
                        raise ValueError(f"Coefficient '{name}' has inconsistent shapes: {self.paramShapes[name]} vs {shape}.")
            
            # Finalize the formula tokens
            reconstructed = ModelFormulaReconstructor.reconstruct(copy.deepcopy(sum))
            print(optimize['type'] + ":", reconstructed)

        if len(self.optimize) == 0:
            raise ValueError("No optimization function found in the formula. Please ensure the formula contains an optimization function.")

        # Initialize coefficients
        self.params = {name: jnp.ones(shape) for name, shape in self.paramShapes.items()}

        # Get global properties
        self.leftPadding = max([formula["leftPadding"] for formula in self.formulas] + 
                               [assignment["leftPadding"] for assignment in self.assignments])
        self.rightPadding = max([formula["rightPadding"] for formula in self.formulas] + 
                                [assignment["rightPadding"] for assignment in self.assignments])

    def requestPadding(self, dataset):
        dataset.setLeftPadding(self.leftPadding)
        dataset.setRightPadding(self.rightPadding)
        self.dataColumns = dataset.columns # Move this to a better place if needed

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

                    valueGetter = GetValueFromData(params, data, self.dataColumns, t0, states = states)
                    newValue = valueGetter.transform(assignment["tokens"].children[1])

                    value = jnp.where(t0 >= assignment["leftPadding"], newValue, oldValue)
                    states[assignment["name"]] = states[assignment["name"]].at[t0, :].set(value)

                for i, formula in enumerate(self.formulas):
                    oldValue = responses[i][t0, :]

                    valueGetter = GetValueFromData(params, data, self.dataColumns, t0, states = states)
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
                    valueGetter = GetValueFromData(params, data, self.dataColumns, t0, statesT0 = states)
                    value = valueGetter.transform(assignment["tokens"].children[1])
                    states[assignment["name"]] = value

                for i, formula in enumerate(self.formulas):
                    valueGetter = GetValueFromData(params, data, self.dataColumns, t0, statesT0 = states)
                    value = valueGetter.transform(formula["tokens"].children[1])
                    responses[i] = value

                return states, responses
            
            batchedForward = jax.vmap(forward)
            states, responses = batchedForward(jnp.arange(dataLength))

        return states, responses

    def loss(self, params, data):
        states, responses = self.predict(params, data)

        losses = []

        for optimize in self.optimize:
            def helper(t0):
                valueGetter = GetValueFromData(params, data, self.dataColumns, t0, states = states)

                if optimize["type"] == "minimize":
                    return valueGetter.transform(optimize["tokens"].children[1])
                elif optimize["type"] == "maximize":
                    return -valueGetter.transform(optimize["tokens"].children[1])
                else:
                    raise ValueError(f"Unsupported optimization type '{optimize['type']}'. Supported types are 'minimize' and 'maximize'.")

            batchedHelper = jax.vmap(helper)
            loss = batchedHelper(jnp.arange(self.leftPadding, data.shape[0] - self.rightPadding))
            lossShape = jnp.shape(loss)
            if len(lossShape) == 1 and lossShape[0] == data.shape[0] - self.leftPadding - self.rightPadding:
                losses.append(loss)
            elif len(lossShape) == 2 and lossShape[0] == data.shape[0] - self.leftPadding - self.rightPadding:
                losses.append(jnp.sum(loss, axis=1))
            else:
                raise ValueError(f"Loss function returned unexpected shape {lossShape}. Expected shape is ({data.shape[0] - self.leftPadding - self.rightPadding},) or ({data.shape[0] - self.leftPadding - self.rightPadding}, n).")

        totalLossPerSample = jax.numpy.sum(jax.numpy.stack(losses, axis=1), axis=1)
        totalLoss = jax.numpy.mean(totalLossPerSample)

        return totalLoss