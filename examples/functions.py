def f():
    pass

import stanza
from stanza import func

a, b = (lambda x: x), (lambda y: y + 1)
a = stanza.func(a)
b = stanza.func(b)
# # Also works on lambdas!
# f_func = stanza.func(f)

# # Testing transpiling mutually recursive functions:
# @func
# def foo():
#     return bar()

# @func
# def bar():
#     return foo()