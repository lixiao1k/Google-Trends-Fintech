import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import xgboost as xgb
import scipy.stats as st
from xgboost.sklearn import XGBRegressor
import matplotlib.pyplot as plt

def getTrainData():
    test = pd.read_csv("train.csv", parse_dates=True, low_memory=False, index_col="Date")
    # test['Year'] = test.index.year
    # test['Month'] = test.index.month
    return test

def getTestData():
    test = pd.read_csv("test.csv", parse_dates=True, low_memory=False, index_col="Date")
    # test['Year'] = test.index.year
    # test["Month"] = test.index.month
    return test


def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat / y-1) ** 2))

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y, yhat)

def aape(yhat, y):
    return np.mean(np.abs((yhat-y)/y))

def aape_xg(yhat, y):
    y = np.exp(y.get_label())
    yhat = np.exp(yhat)
    return "aape", aape(yhat, y)

# def getBestParam():
#     train_data = getTrainData()
#     predictors = [x for x in train_data.columns if x not in ['Seasonally_adjusted_annual_rate', ]]
#     Y = np.log(train_data.Seasonally_adjusted_annual_rate)
#     X = train_data
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
#     # XGB with sklearn wrapper
#     # the same parameters as for xgboost model
#     params_sk = {'max_depth': 10,
#                  'n_estimators': 300,  # the same as num_rounds in xgboost
#                  'objective': 'reg:linear',
#                  'subsample': 0.8,
#                  'colsample_bytree': 0.85,
#                  'learning_rate': 0.1,
#                  'seed': 42}
#
#     skrg = XGBRegressor(**params_sk)
#
#     skrg.fit(X_train, Y_train)
#     params_grid = {
#         'learning_rate': st.uniform(0.01, 0.3),
#         'max_depth': list(range(2, 20, 2)),
#         'gamma': st.uniform(0, 10),
#         'reg_alpha': st.expon(0, 50)}
#
#     search_sk = RandomizedSearchCV(skrg, params_grid, cv=5)  # 5 fold cross validation
#     search_sk.fit(X_train, Y_train)
#
#     # best parameters
#     print(search_sk.best_params_);
#     print(search_sk.best_score_)

def train():
    train_data = getTrainData()
    predictors = [x for x in train_data.columns if x not in ['Seasonally_adjusted_annual_rate', ]]
    Y = np.log(train_data.Seasonally_adjusted_annual_rate)
    X = train_data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    params = {'booster': 'gbtree',
              'objective': 'reg:linear',
              'subsample': 0.5,
              'colsample_bytree': 1,
              'min_child_weight': 9,
              'eta': 0.12,
              'max_depth': 5,
              'seed': 42,
              'lambda': 1}
    dtrain = xgb.DMatrix(X_train[predictors], Y_train)
    dtest = xgb.DMatrix(X_test[predictors], Y_test)

    watchlist = [(dtrain, 'train'), (dtest, 'test')]
    xgb_model = xgb.train(params, dtrain, 72, evals=watchlist, early_stopping_rounds=8, feval=aape_xg, verbose_eval=True)
    xgb.plot_importance(xgb_model)
    plt.show()
    return xgb_model

def test():
    model = train()
    test_data = getTestData()
    predictors = [x for x in test_data.columns if x not in ['Seasonally_adjusted_annual_rate', ]]
    unseen = xgb.DMatrix(test_data[predictors])
    test_p = model.predict(unseen)
    error = aape(np.exp(test_p), test_data.Seasonally_adjusted_annual_rate.values)
    print error


if __name__ == '__main__':
    test()