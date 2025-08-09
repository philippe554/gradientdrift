from lark import Lark
from lark.reconstruct import Reconstructor

formulaGrammar = r"""

    specification: NEWLINE* namedmodel (NEWLINE+ namedmodel)* NEWLINE*
    namedmodel: NAME ":" NEWLINE* model

    model: NEWLINE* statement (NEWLINE+ (statement))* NEWLINE*
    
    ?statement: (parameterdefinition | formula | initialization | assignment | optimize | constraint)

    formula: dependingvariables "~" NEWLINE* sum

    initialization: NAME "[" NEWLINE* NUMBER NEWLINE* "]" "=" NEWLINE* sum
    assignment: sum "=" NEWLINE* sum
    
    parameterdefinition: ("const")? parameterlist ("=" NEWLINE* (sum | shape))+
    parameterlist: parameter ("," NEWLINE* parameter)*
    shape: "[" NEWLINE* (NUMBER NEWLINE* ("," NEWLINE* NUMBER NEWLINE* )*)? "]"

    constraint: boundconstraint | sumconstraint 
    boundconstraint: (sum "<" parameterlist) | (parameterlist "<" sum) | (sum "<" parameterlist "<" sum)
    sumconstraint: ("sum" "(" parameterlist ")" "==" sum) | (sum "==" "sum" "(" parameterlist ")")
    
    optimize: OPTIMIZETYPE ":" sum
    OPTIMIZETYPE: "minimize" | "maximize"

    COMMENT: /#[^\n]*/
    NEWLINE: ( /\r?\n[\t ]*/ | COMMENT )+

    dependingvariables: sum ("," sum)*

    ?sum: product (sumoperator NEWLINE* product)*
    sumoperator: SUMOPERATOR -> operator
    SUMOPERATOR: "+" | "-"

    ?product: exponent (productopertor NEWLINE* exponent)*
    productopertor: PRODUCTOPERATOR -> operator
    PRODUCTOPERATOR: "*" | "/" | "@"

    ?exponent: atom ("^" NEWLINE* atom)?

    ?atom: SIGNED_NUMBER -> number
         | variable
         | parameter
         | funccall
         | "(" NEWLINE* sum NEWLINE* ")"

    funccall: FUNCNAME "(" NEWLINE* [arguments NEWLINE*] ")"
    arguments: sum ("," NEWLINE* sum)*
    parameter: "{" NEWLINE* NAME ("." NAME)? NEWLINE* "}"
    variable: NAME ("." NAME)?

    FUNCNAME: /[a-zA-Z0-9.]+/

    %import common.CNAME -> NAME
    %import common.NUMBER
    %import common.SIGNED_NUMBER
    %import common.WS_INLINE
    %ignore WS_INLINE
"""

parserCache = {}
def getParser(start = 'model'):
    if start not in parserCache:
        parserCache[start] = Lark(formulaGrammar, start = start, maybe_placeholders = False)
    return parserCache[start]


ModelFormulaReconstructor = Reconstructor(getParser())
