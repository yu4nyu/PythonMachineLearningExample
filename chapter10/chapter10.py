### 001 load data
print('### 001')
import pandas as pd
#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
#                header=None, sep='\s+')

df = pd.read_csv('housing.data', header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
             'NOX', 'RM', 'AGE', 'DIS', 'RAD',
             'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
print(df.head())
print()



### 002 visualize the pair-wise correlations between the different features
import matplotlib.pyplot as plt
import seaborn as sns # a Python library for drawing statistical plots based on matplotlib

sns.set(style='whitegrid', context='notebook')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size=2.5)
plt.show()

import numpy as np

# plot the correlation matrix array as a heat map
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)
plt.show()



### 003 solve regression for regression parameters with gradient descent
print('### 003')
from lr_gd import LinearRegressionGD
from sklearn.preprocessing import StandardScaler

X = df[['RM']].values
y = df[['MEDV']].values
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y)
# 2D to 1D, otherwise error will occur from lr_gd.py, when executing "self.w_[1:] += self.eta * X.T.dot(errors)"
# ValueError: non-broadcastable output operand with shape (1,) doesn't match the broadcast shape (1,506)
y_std = y_std.flatten()
lr = LinearRegressionGD()
lr.fit(X_std, y_std)

# plot the cost against the number of epochs to check if the linear regression has converged
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()

# visualize how well the linear regression line fits the training data
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')
    return None

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()

# scale the predicted price outcome back on the Price in $1000's axes
num_rooms_std = sc_x.transform(np.array([5.0]).reshape(1, -1)) # 5 roooms
price_std = lr.predict(num_rooms_std)
print("Price in $1000's: %.3f" % sc_y.inverse_transform(price_std))

print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])
print()



### 004 Estimate the coefficient of a regression model via scikit-learn
print('### 004')
from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X, y)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)
print()

lin_regplot(X, y, slr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()



### 005 fit a robust regression model using RANSAC
from sklearn.linear_model import RANSACRegressor

ransac = RANSACRegressor(LinearRegression(),
            max_trials=100,
            min_samples=50,
            residual_metric=lambda x: np.sum(np.abs(x), axis=1), # DeprecationWarning
            residual_threshold=5.0,
            random_state=0)
ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask], c='blue', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], c='lightgreen', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper left')
plt.show()
