# -*- coding: utf-8 -*-

import theano
import theano.tensor as T


x = T.dvector('x')  # vector
y = T.sum(x ** 2)

dy = T.grad(y, x)

f = theano.function([x], dy)

print f([1.0, 2.0])       # => [2.  4.]
print f([3.0, 4.0, 5.0])  # => [6.  8.  10.]
