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



### 006 logistic function
print('### 006')
X = np.array([[1, 1.4, 1.5]])
w = np.array([0.0, 0.2, 0.4])

def net_input(X, w):
    z = X.dot(w)
    return z

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_activation(X, w):
    z = net_input(X, w)
    return logistic(z)

print('P(y=1|x) = %.3f' % logistic_activation(X, w)[0])

# W : array, shape = [n_output_units, n_hidden_units+1]
#           Weight matrix for hidden layer -> output layer.
# note that first column (A[:][0] = 1) are the bias units
W = np.array([[1.1, 1.2, 1.3, 0.5],
              [0.1, 0.2, 0.4, 0.1],
              [0.2, 0.5, 2.1, 1.9]])
# A : array, shape = [n_hidden+1, n_samples]
#           Activation of hidden layer.
# note that first element (A[0][0] = 1) is the bias unit
A = np.array([[1.0],
              [0.1],
              [0.3],
              [0.7]])
# Z : array, shape = [n_output_units, n_samples]
#           Net input of the output layer.
Z = W.dot(A)
y_probas = logistic(Z)
print('Probabilities:\n', y_probas)

y_class = np.argmax(y_probas, axis=0)
print('predicted class label: %d' % y_class[0])
print()



### 007 estimate probabilities in multi-class classification via the softmax function
print('### 007')

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

def softmax_activation(X, w):
    z = net_input(X, w)
    return softmax(z)

y_probas = softmax(Z)
print('Probabilities:\n', y_probas)
print('Probability sum: %d' % y_probas.sum())

y_class = np.argmax(y_probas, axis=0)
print('predicted class label: %d' % y_class[0])
print()



### 008 broad the output spectrum by using a hyperbolic tangent
import matplotlib.pyplot as plt
from scipy.special import expit

def tanh(z):
    e_p = np.exp(z)
    e_m = np.exp(-z)
    return (e_p - e_m) / (e_p + e_m)

z = np.arange(-5, 5, 0.005)
log_act = logistic(z)
#log_act = expit(z) # the logistic function is available in SciPy's special module
tanh_act = tanh(z)
#tanh_act = np.tanh(z) # we can use NumPy's tanh function to achieve the same results

plt.ylim([-1.5, 1.5])
plt.xlabel('net input $z$')
plt.ylabel('activation $\phi(z)$')
plt.axhline(1, color='black', linestyle='--')
plt.axhline(0.5, color='black', linestyle='--')
plt.axhline(0, color='black', linestyle='--')
plt.axhline(-1, color='black', linestyle='--')

plt.plot(z, tanh_act, linewidth=2, color='black', label='tanh')
plt.plot(z, log_act, linewidth=2, color='lightgreen', label='logistic')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()



### 008 keras
# built on top of Theano. It allows us to utilize our
# GPU to accelerate neural network training
print('### 008')
print('You need to execute the separated script mnist_keras_mlp.py')
print('Using CPU:')
print('    python3 mnist_keras_mlp.py')
print('Using GPU:')
print('    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_keras_mlp.py')
print()
