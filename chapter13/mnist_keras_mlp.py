from load_mnist import load_mnist

# load data
X_train, y_train = load_mnist('mnist', kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
X_test, y_test = load_mnist('mnist', kind='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

# cast the MNIST image array into 32-bit format
import theano
theano.config.floatX = 'float32'
X_train = X_train.astype(theano.config.floatX)
X_test = X_test.astype(theano.config.floatX)

# convert the class labels (integers 0-9) into the one-hot format
# Fortunately, Keras provides a convenient tool for this
from keras.utils import np_utils
print('First 3 labels: ', y_train[:3])
y_train_ohe = np_utils.to_categorical(y_train)
print('\nFirst 3 labels (one-hot):\n', y_train_ohe[:3])


### implement a neural network
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import numpy as np

np.random.seed(1)

model = Sequential() # implement a feedforward neural network
# input layer to a hidden layer
model.add(Dense(input_dim=X_train.shape[1],
                output_dim=50,
                init='uniform',
                activation='tanh'))
# first hidden layer to the second hidden layer.
# two hidden layers with 50 hidden units plus 1 bias unit each.
# Note that bias units are initialized to 0 in fully connected networks
# in iKeras. A common convention is to initialize the bias units to 1.
model.add(Dense(input_dim=50,
                output_dim=50,
                init='uniform',
                activation='tanh'))
# the second hidden layer to output layer
model.add(Dense(input_dim=50,
                output_dim=y_train_ohe.shape[1],
                init='uniform',
                activation='softmax'))

sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
# loss is cost (or loss) function
# The (binary) cross-entropy is just the technical term for the cost function in logistic
# regression, and the categorical cross-entropy is its generalization for multi-class
# predictions via softmax.
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(X_train,
          y_train_ohe,
          nb_epoch=50,
          batch_size=300,
          verbose=1, # to follow the optimization of the cost function during training
          validation_split=0.1)
#          show_accuracy=True) # latest keras removed this parameter

y_train_pred = model.predict_classes(X_train, verbose=0)
print('First 3 predictions: ', y_train_pred[:3])

train_acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print('Training accuracy: %.2f%%' % (train_acc * 100))

y_test_pred = model.predict_classes(X_test, verbose=0)
test_acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print('Test accuracy: %.2f%%' % (test_acc * 100))
