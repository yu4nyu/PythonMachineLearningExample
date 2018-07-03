### 001 serialize fitted scikit-learn estimators
import sys
sys.path.insert(0, '../chapter8/chapter8_part2.py')
exec(open("../chapter8/chapter8_part2.py").read())

import pickle
import os

dest = os.path.join('movieclassifier', 'pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)
# 'wb' means in binary mode to write.
# we set protocol=4 to choose the latest and most efficient pickle protocol that
# has been added to Python 3.4.
pickle.dump(stop, open(os.path.join(dest, 'stopwords.pkl'), 'wb'), protocol=4)
pickle.dump(clf, open(os.path.join(dest, 'classifier.pkl'), 'wb'), protocol=4)
