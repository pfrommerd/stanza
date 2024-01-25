from stanza import _stanza
from stanza._stanza import (
    Function, Ast
)

import dill.source as source
import ast
import types

def parse(f):
    if not isinstance(f, types.FunctionType):
        raise TypeError("Expected function, got {}".format(type(f)))
    # TODO: handle callables that aren't functions or lambdas
    src = source.getsource(f)
    path = source.getsourcefile(f)
    lineno = source.getsourcelines(f)[1]
    # To handle lambdas, do some fancy stuff
    # to get the OG lambda's source code
    # This involves analyzing the line where the lambda is defined,
    # compiling all the lambdas in the line, and comparing the bytecode
    if f.__name__ == "<lambda>":
        source_ast = ast.parse(src, path, type_comments=True)
        lambdas = [node for node in ast.walk(source_ast)
                    if isinstance(node, ast.Lambda)]
        for l in lambdas:
            c = compile(ast.Expression(l), "<string>", "eval")
            # construct the lambda object
            lv = eval(c, {}, {})
            # check if the bytecode is the same as the object we have
            if lv.__code__.co_code == f.__code__.co_code:
                lam = l
                # if it is, we've found the lambda's AST node
                break
        if lam is None:
            raise ValueError("Could not find lambda in source code")
        src = ast.get_source_segment(src, lam)
        # get the tokens associated with the lambda
        return _stanza.parse(src, path, lineno)
    else:
        return _stanza.parse(src, path, lineno)

def jit(f):
    if isinstance(f, Function):
        return f
    if not isinstance(f, types.FunctionType):
        raise TypeError("Expected function, got {}".format(type(f)))
    ast = parse(f)
    expr = _stanza.transpile(ast)
    return _stanza.Function(expr, f.__closure__)