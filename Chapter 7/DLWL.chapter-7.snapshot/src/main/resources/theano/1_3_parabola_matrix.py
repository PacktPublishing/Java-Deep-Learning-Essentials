# -*- coding: utf-8 -*-

import theano
import theano.tensor as T


x = T.dmatrix('x')  # matrix
y = T.sum(x ** 2)

dy = T.grad(y, x)

f = theano.function([x], dy)

print f([[1.0, 2.0],    # => [[2.  4.]
         [3.0, 4.0]])   #      6.  8.]]

print f([[5.0, 6.0],    # => [[10.   12.]
         [7.0, 8.0],    #      14.   16.]
         [9.0, 10.0]])  #      18.   20.]]
