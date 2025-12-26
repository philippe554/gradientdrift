from lark import Lark
from lark.reconstruct import Reconstructor

formulaGrammar = r"""

    specification: NEWLINE* namedmodel (NEWLINE+ namedmodel)* NEWLINE*
    namedmodel: NAME ":" NEWLINE* model

    model: NEWLINE* statement (NEWLINE+ (statement))* NEWLINE*
    
    ?statement: (prior | parameterdefinition | formula | initialization | assignment | optimize | constraint)

    formula: dependingvariables "~" NEWLINE* sum

    initialization: NAME "[" NEWLINE* NUMBER NEWLINE* "]" "=" NEWLINE* sum
    assignment: sum "=" NEWLINE* sum
    
    parameterdefinition: ("const")? ("latent")? parameterlist ("=" NEWLINE* (sum | shape))+
    parameterlist: parameter ("," NEWLINE* parameter)*
    shape: "[" NEWLINE* (NUMBER NEWLINE* ("," NEWLINE* NUMBER NEWLINE* )*)? "]"

    prior: parameter "~" NEWLINE* sum

    constraint: boundconstraint | sumconstraint 
    boundconstraint: (sum "<" parameterlist) | (parameterlist "<" sum) | (sum "<" parameterlist "<" sum)
    sumconstraint: ("sum" "(" parameterlist ")" "==" sum) | (sum "==" "sum" "(" parameterlist ")")
    
    optimize: OPTIMIZETYPE ":" sum
    OPTIMIZETYPE: "minimize" | "maximize"

    COMMENT: /#[^\n]*/
    NEWLINE: ( /\r?\n[\t ]*/ | COMMENT )+

    dependingvariables: sum ("," sum)* -> array

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
         | array
         | "(" NEWLINE* sum NEWLINE* ")"

    funccall: FUNCNAME "(" NEWLINE* [arguments NEWLINE*] ")"
    arguments: sum ("," NEWLINE* sum)*

    array: "{" NEWLINE* (sum ("," NEWLINE* sum)*)? NEWLINE* "}"
    
    parameter: "{" NEWLINE* VALUENAME indexlist? NEWLINE* "}"
    variable: VALUENAME indexlist?

    indexlist: "[" NEWLINE* index ("," NEWLINE* index)* NEWLINE* "]"
    index: NUMBER | VALUENAME | slice
    slice: ( (NUMBER)? ":" (NUMBER)? (":" NUMBER)? )
    
    VALUENAME: /[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)?/
    FUNCNAME: /[a-zA-Z_][a-zA-Z0-9_]*/

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
