### 003 load data from csv file
import numpy as np
import pandas as pd

print('### 003')
df = pd.read_csv('./movie_data.csv')
print(df.head(3))
print()



### 004 transform words into feature vectors
print('### 004')
from sklearn.feature_extraction.text import CountVectorizer

# 1-gram representation is used by default, we could switch to a 2-gram
# representation by initializing a new CountVectorizer instance with
# ngram_range=(2,2).
count = CountVectorizer()
docs = np.array([
       'The sun is shining',
       'The weather is sweet',
       'The sun is shining and the weather is sweet'])
bag = count.fit_transform(docs)
print(count.vocabulary_)
# feature vectors, are also called the raw term frequencies: tf(t,d)
# the number of times a term t occurs in a document d
print(bag.toarray())
print()



### 005 td-idf
print('### 005')
from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer()
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())
print()



### 006 clean text data by regexp
print('### 006')
print(df.loc[0, 'review'][-50:])

import re

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    # Although the addition of the emoticon characters to the end of the cleaned document
    # strings may not look like the most elegant approach, the order of the words doesn't
    # matter in our bag-of-words model if our vocabulary only consists of 1-word tokens.
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

print(preprocessor(df.loc[0, 'review'][-50:]))
print(preprocessor("</a>This :) is :( a test :-)!"))
print()

# now apply our preprocessor function to all movie reviews in our DataFrame
df['review'] = df['review'].apply(preprocessor)



### 007 process documents into tokens
print('### 007')

# One way to tokenize documents is to split them into individual words
# by splitting the cleaned document at its whitespace characters
def tokenizer(text):
    return text.split()

print(tokenizer('runners like running and thus they run'))

# another useful technique is word stemming, which is the process of
# transforming a word into its root form that allows us to map related
# words to the same stem.
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

print(tokenizer_porter('runners like running and thus they run'))
print()



### 008 stop-word removal
print('### 008')

from nltk.corpus import stopwords

stop = stopwords.words('english')
print([w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop])
print()



### 009 train a logistic regression model for document classification
print('### 009')
print('This step will take nearly 2 hours running in a desktop with 16G RAM and E5-2670 CPU')
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# TfidVectorizer is equivalent to CountVectorizer followed by TfidfTransformer.
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
# The first dictionary uses the TfidfVectorizer with its default settings
# (use_idf=True, smooth_idf=True, and norm='l2') to calculate the tf-idfs;
param_grid = [{'vect__ngram_range': [(1,1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1,1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]}
             ]
lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(random_state=0))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
gs_lr_tfidf.fit(X_train, y_train)

print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))
