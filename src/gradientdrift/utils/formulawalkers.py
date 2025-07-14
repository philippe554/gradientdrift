import numpy as np
import re
import copy
import operator as op
import inspect
from lark import Transformer, v_args, Tree, Token
from lark.visitors import Interpreter
import jax.numpy as jnp
from jax import lax
import jax

from gradientdrift.utils.formulaparsers import ModelFormulaReconstructor
from gradientdrift.utils.functionmap import getFunctionMap, requiresRandomKey

# Extract all functions from jax, jax.numpy, and jax.scipy

functionMap = getFunctionMap()

def getParameterName(tree):
    name = ModelFormulaReconstructor.reconstruct(copy.deepcopy(tree))
    s = re.sub(r'[^a-zA-Z0-9_]+', '_', name)
    return s.strip('_')

class RemoveNewlines(Transformer):
    def __default__(self, data, children, meta):
        children = [child for child in children if not (isinstance(child, Token) and child.type == 'NEWLINE')]
        return Tree(data, children, meta)
                
class GetDataFields(Transformer):
    """A transformer to find all unique data fields (variables) used in a formula."""
    def __default__(self, data, children, meta):
        return set().union(*children)
    def variable(self, items): return {items[0].value}
    def number(self, items): return set()
    def parameter(self, items): return set()
    @v_args(inline = True)
    def funccall(self, name, args=None):
        return args if args is not None else set()
    def operator(self, items):
        return set()
    
class SetLeafNodeLags(Interpreter):
    """A visitor to set the lag for leaf nodes in a formula."""
    def __default__(self, tree):
        if isinstance(tree, Tree):
            if not hasattr(tree.meta, 'lag'):
                tree.meta.lag = 0
            for arg in tree.children:
                if isinstance(arg, Tree):
                    arg.meta.lag = tree.meta.lag
                    self.visit(arg)

    @v_args(inline = True, meta = True)
    def funccall(self, meta, name, args=None):
        if not hasattr(meta, 'lag'):
            meta.lag = 0
        if name == 'lag':
            if args and len(args.children) == 2:
                treeNumber = args.children[1]
                if isinstance(treeNumber, Tree) and treeNumber.data == 'number':
                    meta.lag += int(treeNumber.children[0].value)
                else:
                    raise ValueError(f"Invalid lag argument: {treeNumber}")
            else:
                meta.lag += 1

        elif name == 'MA' or name == 'AR':
            if args and len(args.children) == 1:
                treeNumber = args.children[0]
                if isinstance(treeNumber, Tree) and treeNumber.data == 'number':
                    meta.lag += int(treeNumber.children[0].value)
                else:
                    raise ValueError(f"Invalid lag argument: {treeNumber}")
            else:
                raise ValueError(f"Function '{name}' requires a lag argument.")

        for arg in args.children:
            if isinstance(arg, Tree):
                arg.meta.lag = meta.lag
                self.visit(arg)

class GetMaxLag(Transformer):
    """A transformer to find the maximum lag used in a formula."""
    def __default__(self, data, children, meta):
        return max(children)
    @v_args(meta = True)
    def variable(self, meta, children): return meta.lag
    def number(self, children): return 0
    def parameter(self, children): return 0
    def operator(self, children): return 0
    @v_args(inline = True, meta = True)
    def funccall(self, meta, name, args=None):
        funcLag = meta.lag if hasattr(meta, 'lag') else 0
        return max(funcLag, args)
        
class GetParameters(Transformer):
    """A transformer to find all unique column names required by a formula."""
    def __default__(self, data, children, meta):
        return [item for sublist in children for item in sublist]
    def variable(self, items): return []
    def number(self, items): return []
    def parameter(self, items): return [items[0].value] if items else []
    def operator(self, items): return []
    @v_args(inline = True)
    def funccall(self, name, args=None):
        return args if args is not None else []
    
class LabelOuterSum(Interpreter):
    """ A visitor to label the outermost summation in a formula."""
    @v_args(meta = True)
    def sum(self, meta, children):
        meta.outerSum = True

class NameUnnamedParameters(Transformer):
    def __init__(self, existingParameters):
        super().__init__()
        self.existingParameters = set(existingParameters)

    def __default__(self, data, children, meta):
        unnamedIndices = [i for i, tree in enumerate(children) if 
            isinstance(tree, Tree) and tree.data == 'parameter' and len(tree.children) == 0]
        
        if len(unnamedIndices) == 0:
            return Tree(data, children, meta)
        elif len(unnamedIndices) > 1:
            raise ValueError("Multiple unnamed parameters found at the same level. Please ensure each parameter is uniquely named.")
        else:
            if unnamedIndices[0] == 0: # parameter is the first child
                otherChildren = children[2:] # skip the parameter and the operator after it
            else: # parameter is not the first child
                otherChildren = children[:unnamedIndices[0] - 1] + children[unnamedIndices[0] + 1:] # skip the parameter and the operator before it
            
            if len(otherChildren) == 0 or len(otherChildren) != len(children) - 2:
                raise ValueError("Unexpected structure in children. Please ensure the formula is correctly structured.")
            otherTree = Tree(data, otherChildren, meta)
            parameterName = getParameterName(otherTree)
            
            if parameterName in self.existingParameters:
                raise ValueError(f"Parameter name '{parameterName}' already exists. Please ensure all parameters have unique names.")
            self.existingParameters.add(parameterName)
            
            children[unnamedIndices[0]] = Tree("parameter", [Token('NAME', parameterName)], meta)
            return Tree(data, children, meta)

class InsertParameters(Transformer):
    """
    "Compiles" a formula by ensuring every term in the RHS summation is
    explicitly multiplied by a named parameter.
    """

    @v_args(meta = True)
    def sum(self, meta, children):
        if hasattr(meta, 'outerSum') and meta.outerSum:
            insertedItems = []
            for i in range(len(children)):
                if i % 2 == 0:
                    parameterName = getParameterName(children[i])

                    parameter = Tree(
                        Token('RULE', 'parameter'), [
                            Token('NAME', parameterName)
                        ],
                        meta
                    )

                    operator = Tree(
                        "operator", [
                            Token('PRODUCTOPERATOR', '@')
                        ],
                        meta
                    )

                    insertedItems.append(Tree("product", [parameter, operator, children[i]], meta))
                else:
                    if children[i].data == 'operator':
                        insertedItems.append(children[i])
                    else:
                        raise ValueError(f"Unexpected item in sum: {children[i]}")

            return Tree("sum", insertedItems, meta)
        else:
            return Tree("sum", children, meta)
        
    @v_args(meta = True)
    def parameter(self, meta, children):
        if children:
            return Tree("parameter", children, meta)
        else:
            raise ValueError("Unnamed parameter found. Please ensure all parameters are named.")

class GetParameterShapes(Transformer):
    def __init__(self, outputLength):
        super().__init__()
        self.parameterShapes = {}
        self.outputLength = outputLength
    def __default__(self, data, children, meta):
        # check if all children are of the same shape
        if children:
            shapes = [child for child in children]
            uniqueShapes = set(shapes)
            if len(uniqueShapes) == 1:
                return uniqueShapes.pop()
            else:
                raise ValueError("Children have different shapes. Please ensure all children are of the same shape.")
        else:
            raise ValueError("No children found. Please ensure the formula is correctly structured.")
    def variable(self, items): return (1,)
    def number(self, items): return (1,)
    def parameter(self, items): return items[0].value
    @v_args(inline = True)
    def funccall(self, name, args=None):
        if args is None:
            args = []
        if type(args) is not list:
            args = [args]

        if name == "lag":
            return args[0]
        elif name == "MA" or name == "AR":
            lags = int(args[0])
            return (lags, self.outputLength)
        elif name in functionMap:
            for i in range(len(args)):
                if type(args[i]) == str:
                    if args[i] in self.parameterShapes:
                        args[i] = self.parameterShapes[args[i]]
                    else:
                        self.setParameterShape(args[i], (self.outputLength,))
                        args[i] = (self.outputLength,)

            if requiresRandomKey(functionMap[name]):
                args = [jax.random.key(0)] + [jnp.zeros(shape) for shape in args]
            else:
                args = [jnp.zeros(shape) for shape in args]
                    
            return jnp.shape(functionMap[name](*args))
        else:
            raise ValueError(f"Function '{name}' is not supported in this context.")
    def operator(self, children):
        if children[0].value == "+":
            return op.add
        elif children[0].value == "-":
            return op.sub
        elif children[0].value == "*":
            return op.mul
        elif children[0].value == "/":
            return op.truediv
        elif children[0].value == "@":
            return op.matmul
        else:
            raise ValueError(f"Unknown operator: {children[0].value}")
    def setParameterShape(self, parameterName, shape):
        if parameterName in self.parameterShapes:
            if self.parameterShapes[parameterName] != shape:
                raise ValueError(f"Parameter '{parameterName}' has inconsistent shapes: {self.parameterShapes[parameterName]} vs {shape}.")
        else:
            self.parameterShapes[parameterName] = shape
    def aggregate(self, children):
        if len(children) < 3:
            raise ValueError("Aggregate operation requires at least tree children.")
        else:
            leftShape = children[0]
            for i in range(1, len(children), 2):
                operator = children[i]
                rightShape = children[i + 1]
                if operator == op.add or operator == op.sub or operator == op.mul or operator == op.truediv:
                    if type(leftShape) == tuple and type(rightShape) == tuple:
                        leftShape = operator(np.zeros(leftShape), np.zeros(rightShape)).shape
                    elif type(leftShape) == tuple and type(rightShape) == str:
                        name = rightShape
                        rightShape = leftShape
                        self.setParameterShape(name, rightShape)
                        leftShape = operator(np.zeros(leftShape), np.zeros(rightShape)).shape
                    elif type(leftShape) == str and type(rightShape) == tuple:
                        name = leftShape
                        leftShape = rightShape
                        self.setParameterShape(name, leftShape)
                        rightShape = operator(np.zeros(leftShape), np.zeros(rightShape)).shape
                    elif type(leftShape) == str and type(rightShape) == str:
                        raise ValueError(f"Cannot derive shape for two parameters: {leftShape} and {rightShape}.")
                    else:
                        raise ValueError(f"Cannot derive shape, should not happen: {leftShape} and {rightShape}.")
                elif operator == op.matmul:
                    if type(leftShape) == tuple and type(rightShape) == tuple:
                        leftShape = np.matmul(np.zeros(leftShape), np.zeros(rightShape)).shape
                    elif type(leftShape) == tuple and type(rightShape) == str:
                        if len(leftShape) == 1:
                            name = rightShape
                            rightShape = [leftShape[-1], self.outputLength]
                            self.setParameterShape(name, rightShape)
                            leftShape = np.matmul(np.zeros(leftShape), np.zeros(rightShape)).shape
                        else:
                            raise ValueError("Not implemented for more than 1D left shape.")
                    elif type(leftShape) == str and type(rightShape) == tuple:
                        if len(rightShape) == 1:
                            name = leftShape
                            leftShape = [self.outputLength, rightShape[0]]
                            self.setParameterShape(name, leftShape)
                            leftShape = np.matmul(np.zeros(leftShape), np.zeros(rightShape)).shape
                        elif len(rightShape) == 2:
                            name = leftShape
                            leftShape = [self.outputLength, rightShape[0], rightShape[1]]
                            self.setParameterShape(name, leftShape)
                            leftShape = np.tensordot(np.zeros(leftShape), np.zeros(rightShape)).shape
                        else:
                            raise ValueError("Not implemented for more than 2D right shape.")
                    elif type(leftShape) == str and type(rightShape) == str:
                        raise ValueError(f"Cannot derive shape for two parameters: {leftShape} and {rightShape}.")
                    else:
                        raise ValueError(f"Cannot derive shape, should not happen: {leftShape} and {rightShape}.")
                else:
                    raise ValueError(f"Unknown operator: {operator}")                         

            return leftShape
    def product(self, children):
        return self.aggregate(children)
    def sum(self, children):
        return self.aggregate(children)
    def arguments(self, children):
        return children
class GetValueFromData(Transformer):
    """
    A transformer to calculate the numerical value of a formula expression.
    This expects a fully compiled formula with explicit parameters.
    """
    def __init__(self, params = None, constants = None, data = None, 
                 dataColumns = None, t0 = 0, statesT0 = None, states = None,
                 parameterizations = None):
        super().__init__()
        self.params = params
        self.constants = constants
        self.data = data
        self.dataColumns = dataColumns
        self.statesT0 = statesT0
        self.states = states
        self.t0 = t0
        self.parameterizations = parameterizations

        if states and statesT0:
            raise ValueError("Both states and statesT0 are provided. Please provide only one of them.")

    @v_args(meta = True)
    def variable(self, meta, children):
        variableName = children[0].value
        if variableName in self.dataColumns:
            t = self.t0 - meta.lag
            variableIndex = self.dataColumns.index(variableName)
            return self.data[t, variableIndex]
        elif self.statesT0 and variableName in self.statesT0:
            if meta.lag != 0:
                raise ValueError(f"Variable '{variableName}' is defined in states but cannot be accessed with a lag. Please ensure it is defined in the formula without a lag.")
            return self.statesT0[variableName]
        elif self.states and variableName in self.states:
            t = self.t0 - meta.lag
            if variableName in self.states:
                return self.states[variableName][t, :]
        else:
            raise ValueError(f"Variable '{variableName}' not found in data or states. Please ensure it is defined in the formula.")
    
    def number(self, children):
        if children:
            return float(children[0].value)
        else:
            raise ValueError("Number expected but not found in the formula.")
        
    def parameter(self, children):
        if children:
            parameterName = children[0].value
            if parameterName in self.params:
                if self.parameterizations is None:
                    raise ValueError("Parameterizations are not defined. Please ensure parameterizations are set for the model.")
                if parameterName in self.parameterizations:
                    type = self.parameterizations[parameterName]["type"]
                    parameterizationParams = self.parameterizations[parameterName]["parameters"]
                    if type == "bounds":
                        parameterName = parameterizationParams[0]
                        bounds = self.parameterizations[parameterName]["bounds"]
                        if bounds[0] is not None and bounds[1] is not None: # full bounds
                            return jax.nn.sigmoid(self.params[parameterName]) * (bounds[1] - bounds[0]) + bounds[0]
                        elif bounds[0] is not None: # lower bound only
                            return jax.nn.softplus(self.params[parameterName]) + bounds[0]
                        elif bounds[1] is not None: # upper bound only
                            return -jax.nn.softplus(self.params[parameterName]) + bounds[1]
                        else: # no bounds
                            raise ValueError(f"Parameter '{parameterName}' has no bounds defined. Please ensure bounds are set for the parameter.")
                    else:
                        raise ValueError(f"Unknown parameterization type: {type}. Supported types are 'bounds'.")
                else:
                    return self.params[parameterName]
            elif self.constants and parameterName in self.constants:
                return self.constants[parameterName]
            else:
                raise ValueError(f"Parameter '{parameterName}' not found in parameters.")
        else:
            raise ValueError("Unnamed parameter found. Please ensure all parameters are named.")

    def operator(self, children):
        if children[0].value == "+":
            return op.add
        elif children[0].value == "-":
            return op.sub
        elif children[0].value == "*":
            return op.mul
        elif children[0].value == "/":
            return op.truediv
        elif children[0].value == "@":
            return op.matmul
        else:
            raise ValueError(f"Unknown operator: {children[0].value}")

    def aggregate(self, children):
        if len(children) == 1:
            return children[0]
        elif len(children) == 3:
            left = children[0]
            operator = children[1]
            right = children[2]
            if operator == op.matmul:
                if np.shape(left) == tuple():
                    left = np.reshape(left, (1,))
                if np.shape(right) == tuple():
                    right = np.reshape(right, (1,))
                if len(np.shape(left)) == 3:
                    operator = jnp.tensordot
            left = operator(left, right)
            return left
        else:
            left = children[0]
            for i in range(1, len(children), 2):
                operator = children[i]
                right = children[i + 1]
                if operator == op.matmul:
                    if np.shape(left) == tuple():
                        left = np.reshape(left, (1,))
                    if np.shape(right) == tuple():
                        right = np.reshape(right, (1,))
                    if len(np.shape(left)) == 3:
                        operator = jnp.tensordot
                left = operator(left, right)
            return left
    
    def product(self, children):
        return self.aggregate(children)

    def sum(self, children):
        return self.aggregate(children)

    @v_args(inline = True, meta = True)
    def funccall(self, meta, name, args=None):
        if name == "lag":
            return args.children[0] if args else 0
        elif name == "AR":
            if args and len(args.children) == 1:
                lags = int(args.children[0])
                t = self.t0 - meta.lag

                d = lax.dynamic_slice(self.data, 
                    (t, 0), 
                    (lags, len(self.dataColumns)))
                return d
            else:
                raise ValueError("AR function requires a lag argument.")
        elif name in functionMap:
            if args is None:
                args = []
            else:
                args = args.children
            if requiresRandomKey(functionMap[name]):
                return functionMap[name](jax.random.key(0), *args)
            else:
                return functionMap[name](*args)
        else:
            raise ValueError(f"Function '{name}' is not supported in this context.")
        
    