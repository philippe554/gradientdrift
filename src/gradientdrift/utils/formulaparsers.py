from lark import Lark
from lark.reconstruct import Reconstructor

formulaGrammar = r"""

    model: NEWLINE* statement (NEWLINE+ (statement))* NEWLINE*
    
    ?statement: (formula | assignment | optimize | parameterdefinition | constraint)

    formula: dependingvariables "~" NEWLINE* sum

    assignment: NAME "=" NEWLINE* sum
    
    parameterdefinition: parameterlist (PARAMETERDEFINITIONOPERATOR NEWLINE* (sum | shape))+
    PARAMETERDEFINITIONOPERATOR: "=" | "~"
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
    ?sumoperator: SUMOPERATOR -> operator
    SUMOPERATOR: "+" | "-"

    ?product: atom (productopertor NEWLINE* atom)*
    ?productopertor: PRODUCTOPERATOR -> operator
    PRODUCTOPERATOR: "*" | "/" | "@"

    ?atom: NUMBER -> number
         | NAME -> variable
         | parameter
         | funccall
         | "(" NEWLINE* sum NEWLINE* ")"

    funccall: FUNCNAME "(" NEWLINE* [arguments NEWLINE*] ")"
    arguments: sum ("," NEWLINE* sum)*
    parameter: "{" NEWLINE* [NAME NEWLINE*] "}"

    FUNCNAME: /[a-zA-Z0-9.]+/

    %import common.CNAME -> NAME
    %import common.NUMBER
    %import common.WS_INLINE
    %ignore WS_INLINE
"""

ModelFormulaParser = Lark(formulaGrammar, start='model', maybe_placeholders = False)

ModelFormulaReconstructor = Reconstructor(ModelFormulaParser)
