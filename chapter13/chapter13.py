### 001 warm-up exercise
print('### 001')
import theano
from theano import tensor as T

# initialize
x1 = T.scalar()
w1 = T.scalar()
w0 = T.scalar()
z1 = w1 * x1 + w0

# compile
net_input = theano.function(inputs=[w1, x1, w0], outputs=z1)

# execute
print('Net input: %.2f' % net_input(2.0, 1.0, 0.5))
print()



### 002 work with array structures
print('### 002')
import numpy as np
theano.config.floatX = 'float32'

# initialize
# we use the optional name argument(here name='x'), when we execute
#   >>> print(x)
#   x
# if we do not specify the name argument, we'll get
#   >>> print(x)
#   <TensorType(float32, matrix)>
# The TensorType can be accessed via the type method if we specified the name
#   >>> print(x.type())
#   <TensorType(float32, matrix)>
x = T.fmatrix(name='x')
x_sum = T.sum(x, axis=0)

# compile
calc_sum = theano.function(inputs=[x], outputs=x_sum)

# execute (Python list)
ary = [[1, 2, 3], [1, 2, 3]]
print('Column sum:', calc_sum(ary))

# execute (Numpy array)
ary = np.array([[1, 2, 3], [1, 2, 3]], dtype=theano.config.floatX)
print('Column sum:', calc_sum(ary))
print()



### 003 the shared variable
# It allows us to spread large objects (arrays) and grants multiple functions read and write access,
# so that we can also perform updates on those objects after compilation.
print('### 003')

# initialize
x = T.fmatrix('x')
w = theano.shared(np.asarray([[0.0, 0.0, 0.0]], dtype=theano.config.floatX))
z = x.dot(w.T)
update = [[w, w + 1.0]]

# compile
net_input = theano.function(inputs=[x], updates=update, outputs=z)

# execute
data = np.array([[1, 2, 3]], dtype=theano.config.floatX)
for i in range(5):
    print('z%d:' % i, net_input(data))
print()



### 004 the givens variable to insert values into the graph before compiling it
# Using this approach, we can reduce the number of
# transfers from RAM over CPUs to GPUs to speed up learning algorithms that use
# shared variables. If we use the inputs parameter in theano.function , data is
# transferred from the CPU to the GPU multiple times. Using givens , we can keep
# the dataset on the GPU if it fits into its memory
print('### 004')

# initialize
data = np.array([[1, 2, 3]], dtype=theano.config.floatX)
x = T.fmatrix('x')
w = theano.shared(np.asarray([[0.0, 0.0, 0.0]], dtype=theano.config.floatX))
z = x.dot(w.T)
update = [[w, w + 1.0]]

# compile
# the variable 'givens' maps a variable name to the actual Python object
net_input = theano.function(inputs=[], updates=update, givens={x: data}, outputs=z)

# execute
for i in range(5):
    print('z%d:' % i, net_input())
print()



### 005 wrap things up - a linear regression example
X_train = np.asarray([[0.0], [1.0],
                      [2.0], [3.0],
                      [4.0], [5.0],
                      [6.0], [7.0],
                      [8.0], [9.0]],
                    dtype=theano.config.floatX)
y_train = np.asarray([1.0, 1.3,
                      3.1, 2.0,
                      5.0, 6.3,
                      6.6, 7.4,
                      8.0, 9.0],
                    dtype=theano.config.floatX)
# Note that we are using theano.config.floatX when we construct the NumPy
# arrays, so we can optionally toggle back and forth between CPU and GPU
# if we want.

# check the convergence
from linreg import train_linreg
import matplotlib.pyplot as plt
costs, w = train_linreg(X_train, y_train, eta=0.001, epochs=10)
plt.plot(range(1, len(costs)+1), costs)
plt.tight_layout()
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()

# prediction
from linreg import predict_linreg
plt.scatter(X_train, y_train, marker='s', s=50)
plt.plot(range(X_train.shape[0]),
         predict_linreg(X_train, w),
         color='gray',
         marker='o',
         markersize=4,
         linewidth=3)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
