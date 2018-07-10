### 001 load MNIST data
print('### 001')
from load_mnist import load_mnist
X_train, y_train = load_mnist('mnist', kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
X_test, y_test = load_mnist('mnist', kind='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))



### 001_version2
# Optionally, we can save the MNIST image data and labels as CSV files to open them
# in programs that do not support their special byte format. However, we should be
# aware that the CSV file format will take up substantially more space on your local
# drive

# only to be executed once
# np.savetxt('train_img.csv', X_train, fmt='%i', delimiter=',')
# np.savetxt('train_labels.csv', y_train, fmt='%i', delimiter=',')
# np.savetxt('test_img.csv', X_test, fmt='%i', delimiter=',')
# np.savetxt('test_labels.csv', y_test, fmt='%i', delimiter=',')

#X_train = np.genfromtxt('train_img.csv', dtype=int, delimiter=',')
#y_train = np.genfromtxt('train_labels.csv', dtype=int, delimiter=',')
#X_test = np.genfromtxt('test_img.csv', dtype=int, delimiter=',')
#y_test = np.genfromtxt('test_labels.csv', dtype=int, delimiter=',')



### 002 visualize examples
import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(25):
    img = X_train[y_train == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()



### 003
from neuralnet import NeuralNetMLP
nn = NeuralNetMLP(n_output=10,
                  n_features=X_train.shape[1],
                  n_hidden=50,
                  l2=0.1,
                  l1=0.0,
                  epochs=1000,
                  eta=0.001,
                  alpha=0.001,
                  decrease_const=0.00001,
                  shuffle=True,
                  minibatches=50,
                  random_state=1)
nn.fit(X_train, y_train, print_progress=True)
