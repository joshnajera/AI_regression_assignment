import sklearn
import xgboost
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.feature_selection import RFECV
from sklearn.model_selection import LeaveOneOut, train_test_split
import pandas as pd
import numpy
import csv
import matplotlib
from datetime import datetime as dt


FILE = 'C:\git\AI_regression_assignment\Dataset.csv'
TEST_SIZE = .05


def main():
    '''   Performs regression and predicts 01 and 02 values   '''

    # CSV header:   Temperature, Pressure, ThermalConductivity, SoundVelocity, O1, O2
    # Features:     Temperature, Pressure, ThermalConductivity, SoundVelocity
    # Targets:      O1, O2

    dataframe = None
    with open(FILE) as file:
        csv_data = csv.reader(file)
        ####    Note: Not using 'Pressure'
        # dataframe = pd.read_csv(file, usecols=['Temperature', 'ThermalConductivity', 'SoundVelocity', 'O1', 'O2'])
        dataframe = pd.read_csv(file, usecols=['Temperature', 'ThermalConductivity', 'O1', 'O2'])


    num = len(dataframe.columns) - 2
    X = dataframe.as_matrix(columns=dataframe.columns[0:num])
    # y1 = dataframe.O1
    y1 = numpy.asarray(dataframe.O1)
    # y2 = dataframe.O2
    y2 = numpy.asarray(dataframe.O2)


    ####    Was testing out scaling and/or normalizing data--   Did not improve accuracy
    # X = sklearn.preprocessing.normalize(X)
    # y1 = sklearn.preprocessing.scale(y1)
    # y2 = sklearn.preprocessing.scale(y2)

    ####    These aren't as accurate
    # clf = LinearRegression()
    # clf = Lasso()

    ####    More accurate
    # clf = RandomForestRegressor(n_estimators=17, criterion="mse", max_features=None)
    # clf = xgboost.XGBRegressor()
    clf = xgboost.XGBRegressor(n_estimators=800, learning_rate=0.02, max_depth=20)
    print(clf)

    ####    This helps for feature selection
    # clf = RFECV(estimator=clf, step=1)

    score, iterations = 0, 0
    t0 = dt.today().now()

    for i in range(40):
        print('\nTest {}'.format(i+1))
        X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=TEST_SIZE)

        clf.fit(X_train,y_train)

        # clf.fit_transform(X_train, y_train)
        # print('Supporting features: {}'.format(clf.support_))
        # print('Feature rankings: {}'.format(clf.ranking_))

        # clf = AdaBoostRegressor(base_estimator=CLF, n_estimators=100, learning_rate=1)
        # clf.fit(X_train, y_train)


        score += clf.score(X_test, y_test)
        # print("Input: {}".format(X_test))
        # print("Predictions: {}".format(clf.predict(X_test)))
        # print("Actual: {}".format(y_test))
        iterations += 1
        print('Average score: ', score/iterations)

        t1 = dt.today().now()
        print('Time:\t{} seconds'.format(t1-t0))

        # print('Feature importance: {}'.format(clf.feature_importances_))



if __name__ == '__main__':
    main()