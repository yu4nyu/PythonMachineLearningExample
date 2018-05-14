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
