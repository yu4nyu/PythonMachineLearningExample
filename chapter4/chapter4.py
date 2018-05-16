### 001 missing values
print('### 001')
import pandas as pd
from io import StringIO

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
0.0,11.0,12.0,'''
# If you are using Python 2.7, you need
# to convert the string to unicode:
# csv_data = unicode(csv_data)
# StringIO() allows us to read the string assigned to csv_data into a
# pandas DataFrame as if it was a regular CSV file on our hard drive.
df = pd.read_csv(StringIO(csv_data))
print(df)
print()



### 002 return the number of missing values per column 
print('### 002')
print(df.isnull().sum())
print()



### 003 remove missing data
print('### 003')
print('drop rows that have at least one NaN')
print(df.dropna())
print('drop columns that have at least one NaN')
print(df.dropna(axis=1))
print('only drop rows where all columns are NaN')
print(df.dropna(how='all'))
print('drop rows that have not at least 4 non-NaN values')
print(df.dropna(thresh=4))
print('only drop rows where NaN appear in specific columns')
print(df.dropna(subset=['C']))
print()



### 004 replace the missing value by mean value of the entire feature column
print('### 004')
from sklearn.preprocessing import Imputer

# Any data array that is to be transformed needs to
# have the same number of features as the data array that was used to fit the model.
imr = Imputer(missing_values='NaN', strategy='mean', axis=0) # axis=1 for row, other options for the strategy paramter are median or most_frequent
imr = imr.fit(df) # fit() is used to learn the parameters from the training data
imputed_data = imr.transform(df.values) # transform() uses those parameters to transform the data
print(imputed_data)
print()



### 005 catagorical data
print('### 005')
df = pd.DataFrame([\
        ['green', 'M', 10.1, 'class1'],\
        ['red', 'L', 13.5, 'class2'],\
        ['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']
print(df)
print()



### 006 mapping ordinal features
print('### 006')
size_mapping = {
                'XL': 3,
                'L': 2,
                'M': 1}
df['size'] = df['size'].map(size_mapping)
print(df)
print()



### 007 encoding class labels
print('### 007')
import numpy as np

class_mapping = {label:idx for idx,label in
                enumerate(np.unique(df['classlabel']))}
print(class_mapping)
print()



### 008 transform the class labels into integers
print('### 008')
df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)
print()



### 009 convert back to the original string representation
print('### 009')
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print(df)
print()



### 010 LabelEncoder class directly implemented in scikit-learn to achieve the same
print('### 010')
from sklearn.preprocessing import LabelEncoder

class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print(y)
print(class_le.inverse_transform(y))
print()



### 011 use LabelEncoder to encode nominal features
print('### 011')
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:,0])
print(X)
print()
# Although the color values don't come in any particular order, a learning algorithm
# will now assume that green is larger than blue, and red is larger than green. Although
# this assumption is incorrect, the algorithm could still produce useful results.
# However, those results would not be optimal.
# A common workaround for this problem is to use a technique called one-hot encoding.



### 012 use OneHotEncoder to uncode nominal features
print('### 012')
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features=[0]) # [0] means to encode first column in the feature matrix X, here it's the color
# By default, the OneHotEncoder
# returns a sparse matrix when we use the transform method, and we converted the
# sparse matrix representation into a regular (dense) NumPy array for the purposes of
# visualization via the toarray method.
print(ohe.fit_transform(X).toarray())
print()
# Sparse matrices are simply a more efficient
# way of storing large datasets, and one that is supported by many scikit-learn
# functions, which is especially useful if it contains a lot of zeros. To omit the toarray
# step, we could initialize the encoder as OneHotEncoder(...,sparse=False) to return
# a regular NumPy array.



### 013 An even more convenient way to create those dummy features
print('### 013')
# Applied on a DataFrame ,
# the get_dummies method will only convert string columns and leave all other
# columns unchanged
print(pd.get_dummies(df[['price', 'color', 'size']]))
print()



### 014 load the Wine dataset
print('### 014')
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol',
                   'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']
print('Class labels', np.unique(df_wine['Class label']))
print(df_wine.head())
print()



### 015 split dataset into a separate test and training dataset
from sklearn.cross_validation import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values # use variable values to get Numpy array
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



### 016 Normalization
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
# We fit the scaler only once on the training data and use those parameters to transform the test set or any new data point
X_test_norm = mms.transform(X_test)



### 017 Standardization
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
# We fit the scaler only once on the training data and use those parameters to transform the test set or any new data point
X_test_std = stdsc.transform(X_test)



### 018 L1 regularization
print('### 018')
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l1', C=0.1)
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))
print()



### 019
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan',
          'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue',
          'gray', 'indigo', 'orange']
weights, params = [], []
for c in np.arange(-4, 6):
    param = 10**float(c)
    lr = LogisticRegression(penalty='l1', C=param, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(param)
weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column], label=df_wine.columns[column+1], color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
#plt.legend(loc='upper left')
#ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
plt.show()



### 020 SBS and KNN
from sklearn.neighbors import KNeighborsClassifier
from sbs import SBS

knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.xlabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()

print('### 020')
# to see what those five features are that yielded such a good performance
k5 = list(sbs.subsets_[8]) # feature numbers decrease from 13 to 1, so subsets_[8] corresponds to 13-8=5 features
print(df_wine.columns[1:][k5])
# evaluate the performance of the KNN classifier on the original test set
knn.fit(X_train_std, y_train)
print('Training accuracy:', knn.score(X_train_std, y_train))
print('Test accuracy:', knn.score(X_test_std, y_test))
# uset the selected 5-feature subset and see how well KNN performs
knn.fit(X_train_std[:, k5], y_train)
print('Training accuracy:', knn.score(X_train_std[:, k5], y_train))
print('Test accuracy:', knn.score(X_test_std[:, k5], y_test))
print()



### 021 Assess feature importance with random forests
from sklearn.ensemble import RandomForestClassifier

feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f+1, 30, feat_labels[f], importances[indices[f]]))
print()

plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], color='lightblue', align='center')
plt.xticks(range(X_train.shape[1]), feat_labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

# set the threshold to 0.15 to reduce the dataset to the 3 most important features
#X_selected = forest.transform(X_train, threshold=0.15) # threshold is for the feature importance
#print(X_selected.shape)
#print()
# scikit-learn 0.19.1 throws AttributeError: 'RandomForestClassifier' object has no attribute 'transform'
