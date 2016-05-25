# -*- coding: utf-8 -*-

import theano
import theano.tensor as T


x = T.dscalar('x')
y = x ** 2

dy = T.grad(y, x)  # define gradients of y

f = theano.function([x], dy)  # register the function to Theano

print f(1)  # => 2.0
print f(2)  # => 4.0
