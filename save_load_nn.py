import os

import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler

from ann import ANN
from ann_util import serialize, deserialize


# load the iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target


# transform the dataset
xsc = MinMaxScaler(feature_range=(0, 1), copy=True)
xsc.fit(X)

ylb = LabelBinarizer()
ylb.fit(y)


# split the dataset
X_train, X_test, y_train, y_test = train_test_split(xsc.transform(X), y, random_state=0)


# load nn if exists else train
nn_fname = 'models/nn_iris_3000epochs.pickle'
if os.path.exists(nn_fname):
    # load
    print('loading the nn')
    nn = deserialize(nn_fname)
else:
    # train
    print('training the nn')
    nn = ANN([4, 10, 3])

    nn.train(X_train, ylb.transform(y_train), 3000)

    serialize(nn, nn_fname)

# predict
preds = np.array([nn.predict(example) for example in X_test])
y_pred = ylb.inverse_transform(preds)

# evaluate
print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
