### 001 load data from database
import pyprind # visualize the progress and estimate time until completion
import pandas as pd
import os

pbar = pyprind.ProgBar(50000) # 50,000 is the number of documents we were going to read in.
labels = {'pos':1, 'neg':0}
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = './aclImdb/%s/%s' % (s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index=True)
            pbar.update()
df.columns = ['review', 'sentiment']



### 002 process data and save as a CSV file
import numpy as np

# the class labels in the assembled dataset are sorted, we'd better to shuffle it
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('./movie_data.csv', index=False)



### 008 stop-word removal
import nltk

nltk.download('stopwords')
