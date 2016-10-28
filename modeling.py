import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
#from xgboost import plot_tree, plot_importance
from sklearn.metrics import accuracy_score, auc, recall_score, precision_score
from sklearn.metrics import roc_curve, auc
import graphviz

from sklearn.cross_validation import train_test_split
# more
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

def modelfit(alg, dtrain, predictors, target, dtest, useTrainCV=True, cv_folds=3, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)#, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='auc')

    #Predict training set:
    dtest_predictions = alg.predict(dtest[predictors])
    dtest_predprob = alg.predict_proba(dtest[predictors])[:,1]

    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtest[target].values, dtest_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtest[target], dtest_predprob)

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

def grid_search(params_test, train, predictors, target, pos_weight):
    gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=100, max_depth=5,
     min_child_weight=4, gamma=0.1, subsample=0.9, colsample_bytree=0.7, reg_alpha=0.01,
     objective= 'binary:logistic', nthread=8, scale_pos_weight=pos_weight, seed=27),
     param_grid = params_test, scoring='roc_auc',n_jobs=8,iid=False, cv=3, verbose=2)
    gsearch1.fit(train[predictors],train[target])
    print gsearch1.grid_scores_
    print "best results: ", gsearch1.best_params_, gsearch1.best_score_

def main():
    df = pd.read_csv("data/data_with_added_dummies.csv")
    train_and_test, final_test = train_test_split(df, test_size=0.10, stratify=df["y"], random_state=42)
    train, test = train_test_split(train_and_test, test_size=0.10, stratify=train_and_test["y"], random_state=42)
    predictors = ['age', 'education', 'default', 'balance', 'housing',
              'loan', 'duration', 'pdays', 'previous',
               'unknown_education', 'job_admin.', 'job_blue-collar',
              'job_entrepreneur', 'job_housemaid', 'job_management',
              'job_retired', 'job_self-employed', 'job_services', 'job_student',
              'job_technician', 'job_unemployed', 'job_unknown',
              'marital_divorced', 'marital_married', 'marital_single',
              'contact_cellular', 'contact_telephone', 'contact_unknown',
              'poutcome_failure', 'poutcome_other',
              'poutcome_success', 'poutcome_unknown']
    target = "y"
    params_test1 = {
    'max_depth':range(3, 10, 2),
    'min_child_weight':range(4, 10, 2)
    }
    #grid_search(params_test1, train, predictors, target, df["y"].mean())
    params_test3 = {
     'gamma':[i/10.0 for i in range(0,5)]
    }
    #grid_search(params_test3, train, predictors, target, df["y"].mean())
    params_test4 = {
     'subsample':[i/10.0 for i in range(6,10)],
     'colsample_bytree':[i/10.0 for i in range(6,10)]
    }
    #grid_search(params_test4, train, predictors, target, df["y"].mean())
    params_test6 = {
     'reg_alpha':[0, 1e-5, 1e-2,  0.1, 1, 100]
    }
    grid_search(params_test6, train, predictors, target, df["y"].mean())

if __name__ == "__main__":
    main()
