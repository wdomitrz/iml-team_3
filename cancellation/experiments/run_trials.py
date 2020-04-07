import pandas as pd
import numpy as np
import nni

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

PATH_DATA = "./../../data/"

def load_data():
    '''Load dataset, use 20newsgroups dataset'''
    X_train = pd.read_csv(PATH_DATA + "x_train.csv")
    X_test = pd.read_csv(PATH_DATA + "x_test.csv")
    y_train = pd.read_csv(PATH_DATA + "y_train.csv")
    y_test = pd.read_csv(PATH_DATA + "y_test.csv")
    return X_train, X_test, y_train, y_test

def get_model(PARAMS):
    '''Get model according to parameters'''
    model = XGBClassifier()
    model.set_params(**PARAMS)
    return model

def run(X_train, X_test, y_train, y_test, model):
    '''Train model and predict result'''
    model.fit(X_train, y_train)
    score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    nni.report_final_result(score)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    
    # get parameters from tuner
    RECEIVED_PARAMS = nni.get_next_parameter()
    print(RECEIVED_PARAMS)
    PARAMS = XGBClassifier().get_params()
    PARAMS.update(RECEIVED_PARAMS)
    model = get_model(PARAMS)
    run(X_train, X_test, y_train, y_test, model)
