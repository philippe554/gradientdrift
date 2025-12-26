import sympy as sp

# 1. Define all symbols
b0, b1 = sp.symbols('beta_0 beta_1')
i, n = sp.symbols('i n', integer=True)
x = sp.IndexedBase('x')
y = sp.IndexedBase('y')

# --- This part is now generic ---

# 2. Define parameters and loss
params = [b0, b1]  # List of parameters to solve for
loss_rss = sp.Sum((y[i] - (b0 + b1 * x[i]))**2, (i, 1, n))

# 3. Get gradient equations
# We apply expand and doit as before
gradients = [sp.expand(sp.diff(loss_rss, p)).doit() for p in params]

print("--- Gradient Equations ---")
for i, grad in enumerate(gradients):
    print(f"Eq {i+1}: {grad} = 0")

# 4. Programmatically build the A matrix and b vector
A = sp.zeros(len(params), len(params))
b = sp.zeros(len(params), 1)

# This dictionary sets all parameters to 0, to isolate the constant term
param_sub_map = {p: 0 for p in params}

for i, eq in enumerate(gradients):
    for j, p in enumerate(params):
        # A[i,j] = coefficient of parameter j in equation i
        A[i, j] = eq.coeff(p)
    
    # b[i] = -1 * (the constant part of equation i)
    b[i] = -eq.subs(param_sub_map)

print("\n--- Programmatically Built System (A*x = b) ---")
print("Matrix A:")
sp.pprint(A)
print("\nVector b:")
sp.pprint(b)

# 5. Solve the matrix system
# We can use .LUsolve() which is efficient for this
solution_vector = A.LUsolve(b)

# Map the solution vector back to the parameter names
solution = {params[i]: solution_vector[i] for i in range(len(params))}

print("\n--- Final Solution ---")
sp.pprint(solution)

exit()






from gradientdrift.utils.formulaparsers import getParser
from gradientdrift.utils.formulawalkers import *
from gradientdrift.utils.symyhelpers import *
import sympy as sp

def solve_linear_matrix_eq(equation, variable):
    """
    Automatically solves a linear matrix equation of the form A*x + B = 0
    for a MatrixSymbol x.
    """
    print(f"--- Attempting to automatically solve for {variable.name} ---")
    
    # We assume the equation is Eq(f(variable), 0)
    f_beta = equation.lhs
    
    # 1. Expand the expression to make it a simple sum
    f_beta_expanded = sp.expand(f_beta)
    print("Expanded Gradient (f_beta):", f_beta_expanded)
    
    # 2. Find 'A' by differentiating f_beta w.r.t. the variable
    # A = d(f_beta) / d(beta)
    A = sp.diff(f_beta_expanded, variable).doit()
    print("Term 'A' (d(f_beta)/d(beta)):", A)
    
    # 3. Find 'B' by substituting the variable with a ZeroMatrix
    # B = f_beta(beta=0)
    B = f_beta_expanded.subs(variable, 
                             sp.ZeroMatrix(variable.rows, variable.cols))
    print("Term 'B' (f_beta(0)):", B)
    
    # 4. We now have the equation A*beta + B = 0
    # The solution is beta = -Inverse(A) * B
    # We use .simplify() to clean up the final expression (e.g., -(-B) -> B)
    
    solution = sp.simplify(-sp.Inverse(A) * B)
    
    return solution

dataLength = sp.Symbol('dataLength', integer=True)

y = sp.MatrixSymbol('y', dataLength, 1)
X = sp.MatrixSymbol('X', dataLength, 3)
beta = sp.MatrixSymbol('beta', 3, 1)

rss = (y - X * beta).T * (y - X * beta)
print("RSS expression:", rss)

grads = sp.diff(rss, beta)
print("Gradient of RSS w.r.t. beta:", grads)

# zero_matrix = sp.ZeroMatrix(3, 1)
# equation = sp.Eq(grads, zero_matrix)
# print("Equation to solve (set gradient to zero):", equation)

# solution = sp.solve(equation, beta)
# print("Closed-form solution for beta:", solution)

# solution = solve_linear_matrix_eq(equation, beta)
# print("Closed-form solution for beta:", solution)

lhs = grads
rhs = sp.ZeroMatrix(3, 1)

print(lhs, "=", rhs)
lhs = lhs.expand()
print(lhs, "=", rhs)
lhs = lhs + 2 * X.T * y
rhs = rhs + 2 * X.T * y
print(lhs, "=", rhs)



exit()

parser = getParser("sum")

formula = "logpdf(y, W*z + mu, sig) + logpdf(z, 0, 1)"

print("formula in text:", formula)

tree = parser.parse(formula)

larkToSymPy = LarkToSymPy()
expr = larkToSymPy.transform(tree)

print("expr, parsed using Lark converted to SymPy:", expr)
print("expr.expand():", expr.expand())
print("sp.simplify(expr):", sp.simplify(expr))
print("sp.simplify(expr).expand():", sp.simplify(expr).expand())
print("sp.simplify(sp.expand(expr)):", sp.simplify(sp.expand(expr)))

expr = expr.expand()

print("=========================")

z = sp.Symbol('z')
E_expr = take_expectation(expr, z)
print("E_z[expr]:", E_expr)

print("=========================")

e_terms = get_expectation_terms(E_expr, z)
print("E_z terms found:")
for term in e_terms:
    print("     ", term)

print("=========================")

print("sp.simplify(E_expr):", sp.simplify(E_expr))

print("=======================================================")

formula = "logpdf(y, W * z + mu, sigma^2)"

print("formula in text model likelihood:", formula)
tree = parser.parse(formula)
larkToSymPy = LarkToSymPy()
expr = larkToSymPy.transform(tree)
print("expr, parsed using Lark converted to SymPy:", expr)

z = sp.Symbol('z')
hessian = sp.hessian(expr, [z])

print("Hessian of the expression w.r.t. z:", hessian)
print("P = -Hessian =", -hessian)

jacobian = sp.Matrix([sp.diff(expr, z)])

print("Jacobian of the expression w.r.t. z:", jacobian)
print("h = Evaluated at z=0:", jacobian.subs({z: 0}))
print("=========================")