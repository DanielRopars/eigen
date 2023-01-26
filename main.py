"""
===================================================
Faces recognition example using eigenfaces and SVMs
===================================================

The dataset used in this example is a preprocessed excerpt of the
"Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

.. _LFW: http://vis-www.cs.umass.edu/lfw/

"""

from time import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform
import sklearn
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from datasets import load_dataset, Features, ClassLabel, Value

data = pandas.read_parquet("chats_flagged_2021-09.parquet")
#print(data)
bodytrain, bodytest, labeltrain, labeltest = train_test_split(
    data['body'],
    data['label'],
    test_size=0.2,
    stratify=data['label']
)
thing = sklearn.feature_extraction.text.TfidfVectorizer(strip_accents='unicode')
x = thing.fit_transform(bodytrain)
y= labeltrain
reg=LogisticRegression()
reg.fit(x, y)
test = [thing.vocabulary_["gura"]]
test = np.array(test)
t2 = test.reshape(-1,1)
print(reg.predict(thing.transform(["u r a poopyhead", 'You smell like ground up beetles with a side of fries', 'than'])))