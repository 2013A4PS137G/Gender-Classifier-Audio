import os
import numpy as np
from tqdm import tqdm
import pickle
from utils import feature_extractor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


def getX(DATA_PATH):
    file_list = os.listdir(DATA_PATH)
    n = len(file_list)
    feats = np.zeros((n, 20))
    for i in tqdm(range(n), desc='Extracting features', position=0, leave=True):
        feats[i, :] = feature_extractor(DATA_PATH + "/" + file_list[i])
    return feats


# A subset of the Common Voice corpus was selected for training
X_male = getX("data/male")
Y_male = np.zeros((X_male.shape[0]))
X_female = getX("data/female")
Y_female = np.ones((X_female.shape[0]))
X = np.vstack(X_male, X_female)
y = np.concatenate(Y_male, Y_female)

kfold = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
scaler = StandardScaler()
X = scaler.fit_transform(X)
pickle.dump(scaler, open('scaler.pkl', 'wb'))

fold_n = 1
curr_max = 0

for train_index, test_index in kfold.split(X, y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    # Train decision tree model
    clf = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)

    print("Decision Tree")
    print("Accuracy on training set: {:.3f}".format(
        clf.score(X_train, y_train)))
    acc = clf.score(X_test, y_test)
    print("Accuracy on test set: {:.3f}".format(acc))
    if(acc > curr_max):
        pickle.dump(clf, open('model.pkl', 'wb'))

    # Train random forest model
    clf = RandomForestClassifier(
        n_estimators=5, random_state=0).fit(X_train, y_train)
    print("Random Forest")
    print("Accuracy on training set: {:.3f}".format(
        clf.score(X_train, y_train)))
    acc = clf.score(X_test, y_test)
    print("Accuracy on test set: {:.3f}".format(acc))
    if(acc > curr_max):
        pickle.dump(clf, open('model.pkl', 'wb'))

    # Train gradient boosting model
    clf = GradientBoostingClassifier(random_state=0).fit(X_train, y_train)
    print("Gradient Boosting")
    print("Accuracy on training set: {:.3f}".format(
        clf.score(X_train, y_train)))
    acc = clf.score(X_test, y_test)
    print("Accuracy on test set: {:.3f}".format(acc))
    if(acc > curr_max):
        pickle.dump(clf, open('model.pkl', 'wb'))

    # Train support vector machine model
    clf = SVC().fit(X_train, y_train)
    print("Support Vector Machine")
    print("Accuracy on training set: {:.3f}".format(
        clf.score(X_train, y_train)))
    acc = clf.score(X_test, y_test)
    print("Accuracy on test set: {:.3f}".format(acc))
    if(acc > curr_max):
        pickle.dump(clf, open('model.pkl', 'wb'))

    # Train neural network model
    clf = MLPClassifier(random_state=0).fit(X_train, y_train)
    print("Multilayer Perceptron")
    print("Accuracy on training set: {:.3f}".format(
        clf.score(X_train, y_train)))
    acc = clf.score(X_test, y_test)
    print("Accuracy on test set: {:.3f}".format(acc))
    if(acc > curr_max):
        pickle.dump(clf, open('model.pkl', 'wb'))

    fold_n = fold_n + 1
