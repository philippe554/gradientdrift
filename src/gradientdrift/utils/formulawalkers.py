import numpy as np
import re
import copy
import operator as op
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
    
class AddNamespace(Transformer):
    def __init__(self, dataColumns = [], modelNamespace = "model"):
        super().__init__()
        self.dataColumns = dataColumns
        self.modelNamespace = modelNamespace

    @v_args(meta = True)
    def index(self, meta, children):
        data = 'index'
        if children and children[0].type == 'VALUENAME':
            if "." not in children[0].value:
                if children[0].value in self.dataColumns:
                    return Tree(data, [Token('VALUENAME', f"data.{children[0].value}")] + children[1:], meta)
                else:
                    return Tree(data, [Token('VALUENAME', f"{self.modelNamespace}.{children[0].value}")] + children[1:], meta)
            else:
                if self.modelNamespace != "model" and children[0].value.startswith("model."):
                    return Tree(data, [Token('VALUENAME', f"{self.modelNamespace}.{children[0].value[6:]}")] + children[1:], meta)
                else:
                    return Tree(data, children, meta)
        else:
            return Tree(data, children, meta)
        
    @v_args(meta = True)
    def variable(self, meta, children):
        data = 'variable'
        if "." not in children[0].value:
            if children[0].value in self.dataColumns:
                return Tree(data, [Token('VALUENAME', f"data.{children[0].value}")] + children[1:], meta)
            else:
                return Tree(data, [Token('VALUENAME', f"{self.modelNamespace}.{children[0].value}")] + children[1:], meta)
        else:
            if self.modelNamespace != "model" and children[0].value.startswith("model."):
                return Tree(data, [Token('VALUENAME', f"{self.modelNamespace}.{children[0].value[6:]}")] + children[1:], meta)
            else:
                return Tree(data, children, meta)
            
    @v_args(meta = True)
    def parameter(self, meta, children):
        data = 'parameter'
        if "." not in children[0].value:
            return Tree(data, [Token('VALUENAME', f"{self.modelNamespace}.{children[0].value}")] + children[1:], meta)
        else:
            if self.modelNamespace != "model" and children[0].value.startswith("model."):
                return Tree(data, [Token('VALUENAME', f"{self.modelNamespace}.{children[0].value[6:]}")] + children[1:], meta)
            else:
                return Tree(data, children, meta)
    
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
    def funccall(self, meta, name, args):
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

        if args is not None:
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
        if args is None:
            args = 0
        return max(funcLag, args)
        
class GetParameterNames(Transformer):
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
    
class GetVariables(Transformer):
    """A transformer to find all unique variable names used in a formula."""
    def __default__(self, data, children, meta):
        return [item for sublist in children for item in sublist]
    def variable(self, items): return [items[0].value]
    def number(self, items): return []
    def parameter(self, items): return []
    def operator(self, items): return []
    @v_args(inline = True)
    def funccall(self, name, args=None):
        return args if args is not None else []
    
class LabelOuterSum(Interpreter):
    """ A visitor to label the outermost summation in a formula."""
    @v_args(meta = True)
    def sum(self, meta, children):
        meta.outerSum = True

class LabelDataDependencies(Transformer):
    def __init__(self, modelNamespace = "model"):
        super().__init__()
        self.modelNamespace = modelNamespace
    @v_args(meta = True)
    def variable(self, meta, children):
        variableName = children[0].value
        if "." not in variableName:
            raise ValueError(f"Variable '{variableName}' is not fully qualified. Please ensure it includes the namespace.")
        if variableName.startswith("data."):
            tree = Tree(Token("RULE", "variable"), children, meta)
            tree.meta.dataDependency = "constant"
            return tree
        else:
            # TODO: Check if a state can be constant
            tree = Tree(Token("RULE", "variable"), children, meta)
            tree.meta.dataDependency = "state"
            return tree
    @v_args(meta = True)
    def number(self, meta, children): 
        tree = Tree(Token("RULE", "number"), children, meta)
        tree.meta.dataDependency = "constant"
        return tree
    @v_args(meta = True)
    def parameter(self, meta, children):
        parameterName = children[0].value
        if "." not in parameterName:
            raise ValueError(f"Parameter '{parameterName}' is not fully qualified. Please ensure it includes the namespace.")
        parameterNamespace = parameterName.split(".")[0]
        if parameterNamespace == self.modelNamespace:
            tree = Tree(Token("RULE", "parameter"), children, meta)
            tree.meta.dataDependency = "parameter"
            return tree
        else:
            tree = Tree(Token("RULE", "parameter"), children, meta)
            tree.meta.dataDependency = "constant"
            return tree
    @v_args(inline = True, meta = True)
    def funccall(self, meta, name, args=None):
        if args is None:
            args = []
        else:
            args = args.children
        dataDependencyPerArgs = [arg.meta.dataDependency for arg in args]
        uniqueDataDependencies = set(dataDependencyPerArgs)
        if len(uniqueDataDependencies) == 1:
            dataDependency = uniqueDataDependencies.pop()
            if dataDependency == "constant":
                tree = Tree(Token("RULE", "funccall"), [Token('FUNCNAME', name), Tree(Token("RULE", "arguments"), args)], meta)
                tree.meta.dataDependency = "constant"
                return tree
            else:
                tree = Tree(Token("RULE", "funccall"), [Token('FUNCNAME', name), Tree(Token("RULE", "arguments"), args)], meta)
                tree.meta.dataDependency = "nonlinear"
                return tree
        else:
            tree = Tree(Token("RULE", "funccall"), [Token('FUNCNAME', name), Tree(Token("RULE", "arguments"), args)], meta)
            tree.meta.dataDependency = "nonlinear"
            return tree

    @v_args(meta = True)
    def product(self, meta, children):
        values = children[::2]
        numberOfParameters = sum(1 for child in values if child.meta.dataDependency == 'parameter')
        numberOfConstants = sum(1 for child in values if child.meta.dataDependency == 'constant')

        operators = children[1::2]
        uniqueOperators = set(op.children[0].value for op in operators)
        if uniqueOperators != {'*'}:
            tree = Tree(Token("RULE", "product"), children, meta)
            tree.meta.dataDependency = "nonlinear"
            return tree
        
        if numberOfConstants == len(values):
            tree = Tree(Token("RULE", "product"), children, meta)
            tree.meta.dataDependency = "constant"
            return tree
        elif numberOfParameters == 1 and numberOfConstants == len(values) - 1:
            tree = Tree(Token("RULE", "product"), children, meta)
            tree.meta.dataDependency = "linearTerm"
            return tree
        else:
            tree = Tree(Token("RULE", "product"), children, meta)
            tree.meta.dataDependency = "nonlinear"
            return tree
        
    @v_args(meta = True)
    def sum(self, meta, children):
        values = children[::2]  # Get every second child starting from the first
        numberOfParameters = sum(1 for child in values if child.meta.dataDependency == 'parameter')
        numberOfConstants = sum(1 for child in values if child.meta.dataDependency == 'constant')
        numberOfLinearTerms = sum(1 for child in values if child.meta.dataDependency == 'linearTerm')
        numberOfLinearSums = sum(1 for child in values if child.meta.dataDependency == 'linearSum')
        numberOfLinearSumsAndConstants = sum(1 for child in values if child.meta.dataDependency == 'linearSumAndConstant')

        if numberOfConstants == len(values):
            tree = Tree(Token("RULE", "sum"), children, meta)
            tree.meta.dataDependency = "constant"
            return tree
        elif numberOfLinearTerms + numberOfLinearSums == len(values) - numberOfConstants:
            tree = Tree(Token("RULE", "sum"), children, meta)
            tree.meta.dataDependency = "linearSum"
            return tree
        elif numberOfParameters == 1 and numberOfLinearTerms + numberOfLinearSums == len(values) - 1 - numberOfConstants:
            tree = Tree(Token("RULE", "sum"), children, meta)
            tree.meta.dataDependency = "linearSumAndConstant"
            return tree
        elif numberOfLinearSumsAndConstants == 1 and numberOfLinearTerms + numberOfLinearSums == len(values) - 1 - numberOfConstants:
            tree = Tree(Token("RULE", "sum"), children, meta)
            tree.meta.dataDependency = "linearSumAndConstant"
            return tree
        else:
            tree = Tree(Token("RULE", "sum"), children, meta)
            tree.meta.dataDependency = "nonlinear"
            return tree

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
            otherTree = Tree(data, otherChildren)
            parameterName = getParameterName(otherTree)
            
            if parameterName in self.existingParameters:
                raise ValueError(f"Parameter name '{parameterName}' already exists. Please ensure all parameters have unique names.")
            self.existingParameters.add(parameterName)

            children[unnamedIndices[0]] = Tree(Token("RULE", "parameter"), [Token('VALUENAME', parameterName)])
            return Tree(data, children)

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
                            Token('VALUENAME', parameterName)
                        ]
                    )

                    operator = Tree(
                        "operator", [
                            Token('PRODUCTOPERATOR', '@')
                        ]
                    )

                    insertedItems.append(Tree(Token("RULE", "product"), [parameter, operator, children[i]]))
                else:
                    if children[i].data == 'operator':
                        insertedItems.append(children[i])
                    else:
                        raise ValueError(f"Unexpected item in sum: {children[i]}")

            return Tree(Token("RULE", "sum"), insertedItems)
        else:
            return Tree(Token("RULE", "sum"), children)

    @v_args(meta = True)
    def parameter(self, meta, children):
        if children:
            return Tree(Token("RULE", "parameter"), children)
        else:
            raise ValueError("Unnamed parameter found. Please ensure all parameters are named.")

class RemoveMeta(Transformer):
    def __default__(self, data, children, meta):
        return Tree(data, children)

class ExpandFunctions(Transformer):
    """
    Expands functions like diff, AR, MA, etc. to their full form.
    """
    @v_args(inline = True, meta = True)
    def funccall(self, meta, name, args=None):
        if name == "residuals":
            if args is None or len(args.children) != 1:
                raise ValueError("Function 'residuals' requires exactly one argument.")
            residualName = args.children[0].children[0].value
            if "." in residualName:
                namespace = residualName.split(".")[0]
                variable = residualName.split(".")[1]
                data = f"data.{variable}"
                variable = f"{namespace}.{variable}"
            else:
                data = "data." + residualName
                variable = residualName

            return Tree(
                Token("RULE", "sum"),
                [
                    Tree(Token("RULE", "variable"), [Token('VALUENAME', variable)]),
                    Tree("operator", [Token('SUMOPERATOR', '-')]),
                    Tree(Token("RULE", "variable"), [Token('VALUENAME', data)])
                ]
            )
        
        elif name == "diff":
            if args is None or len(args.children) != 1:
                raise ValueError("Function 'diff' requires exactly one argument.")
            
            element = RemoveMeta().transform(args.children[0])

            laggedElement = Tree(
                Token("RULE", "funccall"),
                [
                    Token('FUNCNAME', "lag"),
                    Tree(Token("RULE", "arguments"), [copy.deepcopy(element)])
                ]
            )

            return Tree(
                Token("RULE", "sum"),
                [
                    copy.deepcopy(element),
                    Tree("operator", [Token('SUMOPERATOR', '-')]),
                    laggedElement
                ]
            )

        else:
            return Tree(Token("RULE", "funccall"), [name, args] if args else [name])

class GetParameterShapesAndDimensionLabels(Transformer):
    def __init__(self, dataShape, outputLength):
        super().__init__()
        self.parameterShapes = {}
        self.parameterDimensionLabels = {}
        self.dataShape = dataShape
        self.outputLength = outputLength
    def __default__(self, data, children, meta):
        # do not attempt to infer shape for indexlist or index
        if data == "indexlist" or data == "index":
            return Tree(data, children, meta)
        
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
    def parameter(self, items):
        if len(items) == 1:
            return items[0].value
        elif len(items) == 2:
            # indexed parameter
            parameterName = items[0].value
            indices = items[1].children
            parameterShape = [] # the total shape of the parameter
            parameterShapeSelected = [] # the shape that the index selects
            for i, index in enumerate(indices):
                indexToken = index.children[0]
                if indexToken.type == 'NUMBER':
                    raise NotImplementedError("Numeric indices are not supported for parameter shape inference. Please use named indices.")
                elif indexToken.type == 'VALUENAME':
                    columnName = indexToken.value
                    if columnName.startswith("data."):
                        columnName = columnName[5:]
                    else:
                        raise ValueError(f"Index '{columnName}' does not refer to a data column. Please ensure indices refer to data columns.")
                    
                    if columnName not in self.dataShape:
                        raise ValueError(f"Column '{columnName}' not found in data shape. Please ensure the column exists.")
                    columnUniqueValues = len(self.dataShape[columnName])
                    parameterShape.append(columnUniqueValues)
                    parameterShapeSelected.append(1)

                    self.setParameterDimensionLabels(parameterName, i, self.dataShape[columnName])
                else:
                    raise ValueError(f"Unexpected index type: {indexToken.type}. Please ensure indices are either numbers or parameter names.")

            self.setParameterShape(parameterName, tuple(parameterShape))
            return tuple(parameterShapeSelected)
        else:
            raise ValueError("Parameter with unexpected number of children. Please ensure the formula is correctly structured.")

    @v_args(inline = True)
    def funccall(self, name, args=None):
        if args is None:
            args = []
        if type(args) is not list:
            args = [args]

        if name == "lag" or name == "diff" or name == "residuals":
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
    def setParameterDimensionLabels(self, parameterName, dimension, labels):
        if parameterName not in self.parameterDimensionLabels:
            self.parameterDimensionLabels[parameterName] = {}
        if dimension in self.parameterDimensionLabels[parameterName]:
            if self.parameterDimensionLabels[parameterName][dimension] != labels:
                raise ValueError(f"Parameter '{parameterName}' dimension {dimension} has inconsistent labels: {self.parameterDimensionLabels[parameterName][dimension]} vs {labels}.")
        else:
            self.parameterDimensionLabels[parameterName][dimension] = labels

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
                        leftShape = operator(np.ones(leftShape), np.ones(rightShape)).shape
                    elif type(leftShape) == tuple and type(rightShape) == str:
                        name = rightShape
                        rightShape = leftShape
                        self.setParameterShape(name, rightShape)
                        leftShape = operator(np.ones(leftShape), np.ones(rightShape)).shape
                    elif type(leftShape) == str and type(rightShape) == tuple:
                        name = leftShape
                        leftShape = rightShape
                        self.setParameterShape(name, leftShape)
                        rightShape = operator(np.ones(leftShape), np.ones(rightShape)).shape
                    elif type(leftShape) == str and type(rightShape) == str:
                        raise ValueError(f"Cannot derive shape for two parameters: {leftShape} and {rightShape}.")
                    else:
                        raise ValueError(f"Cannot derive shape, should not happen: {leftShape} and {rightShape}.")
                elif operator == op.matmul:
                    if type(leftShape) == tuple and type(rightShape) == tuple:
                        leftShape = np.matmul(np.ones(leftShape), np.ones(rightShape)).shape
                    elif type(leftShape) == tuple and type(rightShape) == str:
                        if len(leftShape) == 1:
                            name = rightShape
                            rightShape = [leftShape[-1], self.outputLength]
                            self.setParameterShape(name, rightShape)
                            leftShape = np.matmul(np.ones(leftShape), np.ones(rightShape)).shape
                        else:
                            raise ValueError("Not implemented for more than 1D left shape.")
                    elif type(leftShape) == str and type(rightShape) == tuple:
                        if len(rightShape) == 1:
                            name = leftShape
                            leftShape = [self.outputLength, rightShape[0]]
                            self.setParameterShape(name, leftShape)
                            leftShape = np.matmul(np.ones(leftShape), np.ones(rightShape)).shape
                        elif len(rightShape) == 2:
                            name = leftShape
                            leftShape = [self.outputLength, rightShape[0], rightShape[1]]
                            self.setParameterShape(name, leftShape)
                            leftShape = np.tensordot(np.ones(leftShape), np.ones(rightShape)).shape
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
                 dataShape = None, t0 = 0, statesT0 = None, states = None,
                 parameterizations = None, randomKey = None):
        super().__init__()
        self.params = params
        self.constants = constants
        self.data = data
        self.dataShape = dataShape
        self.dataColumns = list(dataShape.keys()) if dataShape else []
        self.statesT0 = statesT0
        self.states = states
        self.t0 = t0
        self.parameterizations = parameterizations
        self.randomKey = randomKey

        if states and statesT0:
            raise ValueError("Both states and statesT0 are provided. Please provide only one of them.")

    @v_args(meta = True)
    def variable(self, meta, children):
        variableName = children[0].value
        if "." not in variableName:
            raise ValueError(f"Variable '{variableName}' is not fully qualified. Please ensure it includes the namespace.")
        if variableName.startswith("data."):
            variableName = variableName[5:]  # Remove 'data.' prefix
            if self.dataColumns and variableName in self.dataColumns:
                t = self.t0 - meta.lag
                variableIndex = self.dataColumns.index(variableName)
                return self.data[t, variableIndex]
            else:
                raise ValueError(f"Variable '{variableName}' not found in data columns. Please ensure it is defined in the formula.")
        else:
            if self.statesT0 and variableName in self.statesT0:
                if meta.lag != 0:
                    raise ValueError(f"Variable '{variableName}' is defined in states but cannot be accessed with a lag. Please ensure it is defined in the formula without a lag.")
                return self.statesT0[variableName]
            elif self.states and variableName in self.states:
                t = self.t0 - meta.lag
                if variableName in self.states:
                    return self.states[variableName][t, :]
                
            raise ValueError(f"Variable '{variableName}' not found in states. Please ensure it is defined in the formula.")
    
    def number(self, children):
        if children:
            if type(children[0].value) == float or type(children[0].value) == int:
                return children[0].value
            else:
                return float(children[0].value)
        else:
            raise ValueError("Number expected but not found in the formula.")
        
    @v_args(meta = True)
    def parameter(self, meta, children):
        if len(children) >= 1:
            parameterName = children[0].value
            if self.params is not None and parameterName in self.params:
                if len(children) == 2:
                    # indexed parameter
                    if parameterName in self.parameterizations:
                        raise NotImplementedError("Indexed parameterizations are not supported yet.")

                    selection = []
                    for index in children[1].children:
                        if index.children[0].type == 'NUMBER':
                            selection.append(int(index.children[0].value))
                        elif index.children[0].type == 'VALUENAME':
                            variableName = index.children[0].value
                            if variableName.startswith("data."):
                                variableName = variableName[5:]
                            if variableName not in self.dataShape:
                                raise ValueError(f"Column '{variableName}' not found in data shape. Please ensure the column exists.")
                            variableIndex = self.dataColumns.index(variableName)
                            t = self.t0 - meta.lag
                            category = self.data[t, variableIndex]
                            category = jnp.astype(category, int)
                            selection.append(category)
                        else:
                            raise ValueError(f"Unexpected index type: {index.children[0].type}. Please ensure indices are either numbers or parameter names.")

                    return self.params[parameterName][tuple(selection)]

                else:
                    # non-indexed parameter
                    if self.parameterizations is None:
                        raise ValueError("Parameterizations are not defined. Please ensure parameterizations are set for the model.")
                    if parameterName in self.parameterizations:
                        unconstrainedParamNames = self.parameterizations[parameterName]["unconstraintParameterNames"]
                        unconstrainedParams = [self.params[unconstrainedParamName] for unconstrainedParamName in unconstrainedParamNames]
                        return self.parameterizations[parameterName]["apply"](*unconstrainedParams)
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

    def exponent(self, children):
        if len(children) == 2:
            base = children[0]
            exponent = children[1]
            if np.shape(base) == tuple():
                base = np.reshape(base, (1,))
            if np.shape(exponent) == tuple():
                exponent = np.reshape(exponent, (1,))
            return jnp.power(base, exponent)
        else:
            raise ValueError("Exponentiation requires exactly two children: base and exponent.")

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
                if self.randomKey is None:
                    raise ValueError(f"Function '{name}' requires a random key, but none was provided.")
                self.randomKey, subkey = jax.random.split(self.randomKey)
                return functionMap[name](subkey, *args)
            else:
                return functionMap[name](*args)
        else:
            raise ValueError(f"Function '{name}' is not supported in this context.")
        
class GetOLSTerms(Interpreter):
    def __init__(self, dataShape):
        self.dataShape = dataShape
        self.terms = {}
        self.bias = None
        self.biasIsPositive = True
        self.constants = []
        super().__init__()

    def __default__(self, tree):
        raise ValueError(f"Unexpected tree structure: {tree}")

    @v_args(meta = True)
    def parameter(self, meta, children):
        cumulativeSignIsPositive = True
        if hasattr(meta, 'cumulativeSignIsPositive'):
            cumulativeSignIsPositive = meta.cumulativeSignIsPositive

        if len(children) == 1:
            # no indexing, so this is a bias term
            parameterName = children[0].value
            if self.bias is not None:
                raise ValueError(f"Multiple biases.")
            else:
                self.bias = parameterName
                self.biasIsPositive = cumulativeSignIsPositive
        elif len(children) == 2:
            # indexed parameter, so this is a term
            parameterName = children[0].value
            if parameterName in self.terms:
                raise ValueError(f"Multiple terms for parameter '{parameterName}'. Please ensure each parameter appears only once in the formula.")
            
            indices = children[1].children
            # currently only support one index
            if len(indices) != 1:
                raise ValueError("Currently only single index parameters are supported.")
            
            indexToken = indices[0].children[0]
            if indexToken.type != 'VALUENAME':
                raise ValueError(f"Unexpected index type: {indexToken.type}. Please ensure indices are parameter names.")
            
            indexName = indexToken.value
            if indexName.startswith("data."):
                indexName = indexName[5:]
                
            if indexName not in self.dataShape:
                raise ValueError(f"Index '{indexName}' not found in data shape. Please ensure the index exists.")
            
            if type(self.dataShape[indexName]) is not list:
                raise ValueError(f"Index '{indexName}' is not categorical. Please ensure the index is categorical.")

            numberOfCategories = len(self.dataShape[indexName])

            oneHotEncoding = Tree("funccall", [
                Token('FUNCNAME', 'one_hot'),
                Tree("arguments", [
                    Tree("variable", [Token('VALUENAME', "data." + indexName)], meta),
                    Tree("number", [Token('SIGNED_NUMBER', numberOfCategories)], meta)
                ])
            ], meta)

            if cumulativeSignIsPositive:
                self.terms[parameterName] = oneHotEncoding
            else:
                self.terms[parameterName] = Tree(Token("RULE", "product"), [
                    Tree(Token("RULE", "number"), [Token("SIGNED_NUMBER", "-1")]),
                    Tree(Token("RULE", "operator"), [Token("PRODUCTOPERATOR", "*")]),
                    oneHotEncoding
                ])

        else:
            raise ValueError("Parameter with unexpected number of children. Please ensure the formula is correctly structured.")

    @v_args(meta = True)
    def product(self, meta, children):
        cumulativeSignIsPositive = True
        if hasattr(meta, 'cumulativeSignIsPositive'):
            cumulativeSignIsPositive = meta.cumulativeSignIsPositive

        if meta.dataDependency == 'linearTerm':
            operators = [o.children[0].value for o in children[1::2]]
            if set(operators) != {"*"}:
                raise ValueError(f"Expected all operators to be '*', but found: {operators}")
            parameter = [c for c in children[::2] if c.data == 'parameter']
            variables = [c for c in children[::2] if c.data != 'parameter']
            if len(parameter) != 1:
                raise ValueError(f"Expected exactly one parameter in term, but found: {len(parameter)}")
            if len(variables) == 0:
                raise ValueError(f"Expected at least one variable in term, but found: {len(variables)}")
            
            parameterName = parameter[0].children[0].value
            if len(variables) == 1:
                if cumulativeSignIsPositive:
                    self.terms[parameterName] = variables[0]
                else:
                    self.terms[parameterName] = Tree(Token("RULE", "product"), [
                        Tree(Token("RULE", "number"), [Token("SIGNED_NUMBER", "-1")]),
                        Tree(Token("RULE", "operator"), [Token("PRODUCTOPERATOR", "*")]),
                        variables[0]
                    ])
            else:
                variablesJoined = [variables[0]]
                for c in variables[1:]:
                    variablesJoined.append(Tree("operator", [Token("PRODUCTOPERATOR", "*")]))
                    variablesJoined.append(c)
                if not cumulativeSignIsPositive:
                    variablesJoined.append(Tree("operator", [Token("PRODUCTOPERATOR", "*")]))
                    variablesJoined.append(Tree(Token("RULE", "number"), [Token("SIGNED_NUMBER", "-1")]))

                self.terms[parameterName] = Tree(Token("RULE", "product"), variablesJoined, meta)
        else:
            raise ValueError(f"Product: Expected linear term, but found: {meta.dataDependency}")

    @v_args(meta = True)
    def sum(self, meta, children):
        outerSignIsPositive = True
        if hasattr(meta, 'cumulativeSignIsPositive'):
            outerSignIsPositive = meta.cumulativeSignIsPositive

        if meta.dataDependency == 'linearSum' or meta.dataDependency == 'linearSumAndConstant':
            for i in range(0, len(children), 2):
                elementSignIsPositive = True if i == 0 else children[i - 1].children[0].value == '+'
                cumulativeSignIsPositive = outerSignIsPositive == elementSignIsPositive 

                if children[i].meta.dataDependency == "constant":
                    if cumulativeSignIsPositive:
                        self.constants.append(children[i])
                    else:
                        self.constants.append(
                            Tree(Token("RULE", "product"), [
                                Tree(Token("RULE", "number"), [Token("SIGNED_NUMBER", "-1")]),
                                Tree(Token("RULE", "operator"), [Token("PRODUCTOPERATOR", "*")]),
                                children[i]
                            ])
                        )
                else:
                    children[i].meta.cumulativeSignIsPositive = cumulativeSignIsPositive
                    self.visit(children[i])
        else:
            raise ValueError(f"Sum: Expected linear term, but found: {meta.dataDependency}")