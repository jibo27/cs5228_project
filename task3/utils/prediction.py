import os
import csv
from numpy.core.numeric import normalize_axis_tuple
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import numpy as np
from datetime import datetime
import pickle
import sklearn

from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

np.random.seed(1)


# print(next(reader_train))
def parse(filename):
    reader = csv.reader(open(filename, 'r', encoding='utf8'))

    attr = next(reader)
    data = []

    for _, line in enumerate(reader):
        d = {}
        for idx in range(len(line)):
            d[attr[idx]] = line[idx]
        data.append(d)

    return data

def self_split(X, y, rate):
    num = X.shape[0]
    n_test = int(num * rate)
    X_train, X_test = X[:-n_test], X[-n_test:]
    y_train, y_test = y[:-n_test], y[-n_test:]
    return X_train, X_test, y_train, y_test

# def get_vector(d):
#     vector = []
#     for k, v in d.items():
#         if k == 

# attr2vec = {} 
# This is a dictionary, where the key is the attribute name
# Each element is also a dictionary, where the key is "value2idx" and "onehot_matrix"
# value2idx is to map an attribute value to the index
# With the index from value2idx, we can use the onehot_matrix[idx] to get the corresponding vector for that value
def date2value(s):
    if len(s) == 0:
        value = -1
    else:
        # value = datetime.strptime(s, '%d-%b-%Y').toordinal()
        value = int(s[-4:])
    return value 

attr_ignored = ['listing_id', 'title', 'description', 'features', 'accessories',
                # 'model', 
                'no_of_owners',  
                # 'original_reg_date',
                 'opc_scheme', 'category']
# attr_ignored = ['listing_id', 'title', 'description', 'features',
#                 'model', 'original_reg_date']


def get_nominal_matrix(values):
    s = set(values)
    s.add('')
    k = {}
    idx2value = list(s)
    value2idx = {value:idx for idx, value in enumerate(idx2value)}
    arr = np.asarray([[v] for v in idx2value])

    encoder = OneHotEncoder(sparse=False)
    onehot_matrix = encoder.fit_transform(arr)

    value2vec = {value:onehot_matrix[idx] for value, idx in value2idx.items()}
    return value2vec


def analyze_attribute(data):
    attrs = list(data[0].keys())
    attr2vec = {}
    data_cleaned  =[]
    for key in attrs:
        if key in attr_ignored:
            continue
        if key == 'price':
            attr2vec[key] = {}
            attr2vec[key] = lambda x: float(x.strip())

        set_attr = set()
        for elm in data:
            # Special consideration for some keys...
            if key == 'make' and elm[key] == '':
                elm[key] = elm['title'].split(' ')[0].lower()
                # print('add %s'%elm[key]
            if key == 'original_reg_date' and elm[key] == '':
                elm['original_reg_date'] = elm['reg_date']



            # if key in ['original_reg_date', 'reg_date', 'lifespan' ]:
            if key in ['reg_date', 'lifespan', 'original_reg_date']:
                attr2vec[key] = date2value
            elif key in ['curb_weight', 'power', 'engine_cap', \
                         'depreciation', 'coe', 'road_tax', \
                         'dereg_value', 'mileage', 'omv', \
                         'arf']: # ratio
                attr2vec[key] = lambda x: float(x.strip()) if len(x.strip()) != 0 else -1
            else:
                value = elm[key].strip()
                set_attr.add(value.lower())
        if 0 < len(set_attr) < 700: # If one attribute only has a small number of value set, we index them
            attr2vec[key] = get_nominal_matrix(set_attr)
            # print('%s is added as a nominal, whose size is %d'%(key, len(set_attr)))
        elif key in attr2vec:
            pass
            # print('Attribute "%s" is added as a function'%(key))
        else:
            print(key, len(set_attr))
            raise ValueError
    return attrs, attr2vec

def get_vector(d, attr2vec, attrs, has_label):
    """
        attrs is a list of attributes excluding the price. It is used to order the vector
    """
    vector = []
    for attr in attrs:
        if not has_label and attr == 'price':
            continue
        if attr in attr_ignored:
            continue
        value = d[attr]

        # Special consideration
        if attr == 'make' and value == '':
            value = d['title'].split(' ')[0].lower()
        if attr == 'original_reg_date' and value == '':
            value = d['reg_date']

        if attr in attr2vec:

            if hasattr(attr2vec[attr], 'shape') or isinstance(attr2vec[attr], dict): # 2 ways of indexing...
                if value not in attr2vec[attr]:
                    value = ''
                vec = attr2vec[attr][value]
            else:
                vec = attr2vec[attr](value)
        if vec is None:
            print(attr, value)

        if isinstance(vec, list) or isinstance(vec, np.ndarray):
            vector += [*vec]
        else:
            vector += [vec]
    return vector

def build_vectors(data, attr2vec, attrs, has_label=True):
    vectors = []
    for idx, elm in enumerate(data):
        vector = get_vector(elm, attr2vec, attrs, has_label)
        vectors.append(vector)
    return np.float32(vectors)

# data_train = parse('../data/train.csv')

# # print(data_train[0].keys())

# attrs, attr2vec = analyze_attribute(data_train)
# # print(attrs)
# # print(attr2vec['make'])

# data_train = build_vectors(data_train, attr2vec, attrs)
# # print(data_train.shape)
# # print(data_train)

# # print()
# X = data_train[:, :-1]
# y = data_train[:, -1]


# # Cross-Validation
# test_ratio = 0.3
# X_train, X_test, y_train, y_test = self_split(X, y, test_ratio)

# other_params ={"n_estimators": 400, "max_depth": 20, "min_child_weight": 2, "colsample_bytree": 0.5}
# learning_rate = 0.09
# model = XGBRegressor(**other_params, learning_rate=learning_rate)
# # print(model)
# model.fit(X_train, y_train)
# # model.fit(X, y)

# def write_to_csvfile(predicted, filename):
#     with open(filename, 'w', newline='') as csvfile:
#         fieldnames = ['Id', 'Predicted']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',')
#         writer.writeheader()
#         # for i in range(y_pred.shape[0]):
#         for i in range(predicted.shape[0]):
#             writer.writerow({'Id': i, 'Predicted': '%.1f'%predicted[i]})
#         csvfile.close()
def write_to_csvfile(predicted, filename, ratio):
    df = pd.read_csv(filename)
    print(predicted.shape)
    print(df.shape)
    num = int(df.shape[0] * ratio)
    df = df[-num:]
    df['market_price'] = predicted
    df.to_csv('data/task3.csv')



# # # validate on the validation set
# y_pred_test = model.predict(X_test)
# # # print('Accuracy on the trainset:', np.linalg.norm(y_pred_test - y_test))
# # print('Accuracy on the cross validation set:', np.linalg.norm(y_test - y_pred_test))
# write_to_csvfile(y_pred_test, '../data/train.csv', ratio=test_ratio)


# # Evaluting...
# data_test = parse('data/test.csv')
# print(data_test[0].keys())
# data_test = build_vectors(data_test, attr2vec, attrs, has_label=False)
# print(data_test.shape)
# y_pred = model.predict(data_test)

# print('Mean of test set:', y_pred.mean())
# write_to_csvfile(y_pred, 'results/submission.csv')