import pandas as pd
import numpy as np
import calendar
import pickle 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

PATH_DATA = "./../data/"

def get_integer_mapping(le):
    '''
    Return a dict mapping labels to their integer values.
    
    le: a fitted sklearn LabelEncoder
    '''
    res = {}
    for cl in le.classes_:
        res.update({cl:le.transform([cl])[0]})

    return res

def prepare_dataset(df):
    df = df.fillna(0)
    month_name_to_num = {name: num for num, name in enumerate(calendar.month_name) if num}
    df['arrival_date_month'] = [month_name_to_num[x] for x in df['arrival_date_month']]
    df["arrival_weekday"] = [calendar.weekday(df.loc[i, 'arrival_date_year'], df.loc[i, 'arrival_date_month'], \
        df.loc[i, 'arrival_date_day_of_month']) for i in df.index]
    df = df.drop(["country", "agent", "company", "reservation_status_date", "reservation_status", \
        "arrival_date_year", "assigned_room_type", "reserved_room_type"], axis=1)
    print(df.head())
    feature_type = df.dtypes
    object_features = [i for i in feature_type.index if feature_type[i] == 'object']
    dict_trans = dict()
    for feat in object_features:
        le = LabelEncoder()
        df[feat] = le.fit_transform(df[feat]) 
        integerMapping = get_integer_mapping(le)
        dict_trans[feat] = integerMapping
    with open(PATH_DATA + 'feat_trans.pkl', 'wb') as f:
        pickle.dump(dict_trans, f, pickle.HIGHEST_PROTOCOL)
    X = df.drop("is_canceled", axis=1)
    y = df.loc[:, 'is_canceled']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)
    X_train.to_csv(PATH_DATA + "x_train.csv", index=False)
    X_test.to_csv(PATH_DATA + "x_test.csv", index=False)
    pd.DataFrame(y_train).to_csv(PATH_DATA + "y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv(PATH_DATA + "y_test.csv", index=False)
    print("Dane zostaly zapisane w katalogu {}.".format(PATH_DATA))
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = pd.read_csv(PATH_DATA + "hotel_bookings.csv")
    prepare_dataset(df)
