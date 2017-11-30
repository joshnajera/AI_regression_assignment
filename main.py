#!/usr/lib/python3
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.feature_selection import RFECV
from datetime import datetime as dt
import pandas as pd
import matplotlib
import sklearn
import xgboost
import numpy
import csv

FILE = 'Dataset.csv'
TEST_SIZE = .25


def main():
    '''   Performs regression and predicts 01 and 02 values   '''

    # CSV header:   Temperature, Pressure, ThermalConductivity, SoundVelocity, O1, O2
    # Features:     Temperature, Pressure, ThermalConductivity, SoundVelocity
    # Targets:      O1, O2

    dataframe = None
    with open(FILE) as file:
        dataframe = pd.read_csv(file, usecols=['Temperature', 'ThermalConductivity', 'O1', 'O2'])
        ####    Note: Not using 'Pressure' or 'SoundVelocity' -- This is because they are creating noise.
	####		I found this out through use of RFECV, a tool to aid in Feature Selection

    ####    'num' used to calculate which columns to use from dataframe
    ####    (we are subtracting 2 because we are predicting two things, O1 and O2)
    num = len(dataframe.columns) - 2
    X = dataframe.as_matrix(columns=dataframe.columns[0:num])
    y1 = numpy.asarray(dataframe.O1)
    y2 = numpy.asarray(dataframe.O2)
    ####    Old way I was grabbing data, not as nice to use when displaying output-- Use numpy variation abouve
    # y1 = dataframe.O1
    # y2 = dataframe.O2


    ####    Was testing out scaling and/or normalizing data--   Did not improve accuracy
    # X = sklearn.preprocessing.normalize(X)
    # y1 = sklearn.preprocessing.scale(y1)
    # y2 = sklearn.preprocessing.scale(y2)


    ####    These aren't as accurate
    # clf = LinearRegression()
    # clf = Lasso()
    ####    More accurate algorithms
    # clf = AdaBoostRegressor(base_estimator=CLF, n_estimators=100, learning_rate=1)
    # clf = RandomForestRegressor(n_estimators=17, criterion="mse", max_features=None)
    clf = xgboost.XGBRegressor(n_estimators=800, learning_rate=0.1, max_depth=30)


    ####    This helps for feature selection
    # clf = RFECV(estimator=clf, step=1)


    print("Classifier being used: {}".format(clf))
    ####    For scoring and timing
    score, iterations = 0, 0
    t0 = dt.today().now()


    for i in range(40):
	####	Following 3 lines are used for Feature Selection process only
        # clf.fit_transform(X_train, y_train)
        # print('Supporting features: {}'.format(clf.support_))
        # print('Feature rankings: {}'.format(clf.ranking_))

        print('\nTest {}'.format(i+1))
        X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=TEST_SIZE)
        clf.fit(X_train,y_train)

        score += clf.score(X_test, y_test)
        iterations += 1
        t1 = dt.today().now()

        # print("Input: {}".format(X_test))
        # print("Predictions: {}".format(clf.predict(X_test)))
        # print("Actual: {}".format(y_test))
        print('Average Score: ', score/iterations)
        print('Total Time:\t{} seconds'.format(t1-t0))
        # print('Feature importance: {}'.format(clf.feature_importances_))


if __name__ == '__main__':
    main()
