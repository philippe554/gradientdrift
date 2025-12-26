
import sympy as sp

def E_op(expression, variable):
    """
    Creates a symbolic representation of the Expectation operator.
    E.g., E_op(x, z) -> E_z(x)
    """
    return sp.Function(f'E_{variable.name}')(expression)

def _expect_helper(expr, z):
    """
    The recursive "transformer" function that walks the expression tree
    of an *already expanded* expression.
    """
    # 1. Base Case: If the term has no z, it's a constant. E[c] = c
    if not expr.has(z):
        return expr

    # 2. Linearity (Add): E[a + b] = E[a] + E[b]
    # We recursively call the helper on each term of the sum.
    if expr.is_Add:
        return sp.Add(*(_expect_helper(arg, z) for arg in expr.args))
            
    # 3. Factoring Constants (Mul): E[c*f(z)] = c*E[f(z)]
    if expr.is_Mul:
        constants = []
        z_parts = []
        
        for arg in expr.args:
            if arg.has(z):
                z_parts.append(arg)
            else:
                constants.append(arg)

        # Reconstruct the two parts
        const_part = sp.Mul(*constants) if constants else sp.Integer(1)
        z_part = sp.Mul(*z_parts)

        # Apply the rule: const * E[f(z)]
        # We wrap the z-dependent part in our expectation operator.
        return const_part * E_op(z_part, z)

    # 4. Atomic/Function Case: E[f(z)]
    # This catches anything that wasn't a sum or product,
    # such as z itself, z**2, sin(z), etc.
    return E_op(expr, z)
        
def take_expectation(expr, z):
    """
    Takes the expectation of a SymPy expression with respect to variable z.
    
    1. Expands the expression to ensure linearity.
    2. Applies the expectation operator recursively.
    """
    # Step 1: Expand the expression. This is crucial.
    expanded_expr = expr.expand()
    
    # Step 2: Apply the recursive helper to the expanded expression.
    return _expect_helper(expanded_expr, z)


def get_expectation_terms(expr, z):
    """
    Finds all unique E_z(...) sub-expressions in the expression.
    
    Args:
        expr: The SymPy expression (output of take_expectation).
        z: The SymPy symbol for the variable of expectation.

    Returns:
        A list of all unique E_z(...) SymPy expressions.
    """
    # 1. Define the *type* (class) of function we are looking for.
    # SymPy's sp.Function('name') creates a class.
    E_z_type = sp.Function(f'E_{z.name}')
    
    # 2. Use .atoms() to find all instances (atoms) of this class.
    # This returns a set, ensuring uniqueness.
    e_terms_set = expr.atoms(E_z_type)
    
    return list(e_terms_set)

def get_required_expectations(expr, z):
    """
    Finds all unique *arguments* of the E_z(...) terms.
    This gives you the list of expressions you need to calculate.

    Args:
        expr: The SymPy expression (output of take_expectation).
        z: The SymPy symbol for the variable of expectation.

    Returns:
        A list of the unique inner expressions (e.g., [z, z**2]).
    """
    # 1. Get all the full E_z(...) terms
    e_terms = get_expectation_terms(expr, z)
    
    # 2. Extract the first (and only) argument from each term
    required_exprs = [term.args[0] for term in e_terms]
    
    return required_exprs