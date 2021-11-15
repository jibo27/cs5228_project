import os
import csv
from numpy.core.numeric import normalize_axis_tuple
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import numpy as np
from datetime import datetime
import pickle

from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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

# def get_vector(d):
#     vector = []
#     for k, v in d.items():
#         if k == 

# nominal2value = {} 
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
                'model', 'original_reg_date', 'opc_scheme', 'category']
# attr_ignored = ['listing_id', 'title', 'description', 'features',
#                 'model', 'original_reg_date']

def analyze_attribute(data):
    attrs = list(data[0].keys())
    nominal2value = {}
    for key in attrs:
        if key in attr_ignored:
            continue
        if key == 'price':
            nominal2value[key] = {}
            nominal2value[key]['value2vec'] = lambda x: float(x.strip())

        set_attr = set()
        for elm in data:
            # if key in ['original_reg_date', 'reg_date', 'lifespan' ]:
            if key in ['reg_date', 'lifespan' ]:
                nominal2value[key] = {}
                nominal2value[key]['value2vec'] = date2value
            elif key in ['curb_weight', 'power', 'engine_cap', \
                         'depreciation', 'coe', 'road_tax', \
                         'dereg_value', 'mileage', 'omv', \
                         'arf']: # ratio
                nominal2value[key] = {}
                nominal2value[key]['value2vec'] = lambda x: float(x.strip()) if len(x.strip()) != 0 else -1
            else:
                value = elm[key].strip()
                set_attr.add(value)
        if key in ['category', 'accessories']:
            print()
            set_attr.add('') # For unseen data
            list_attr = [a1.split(',') for a1 in set_attr]
            set_attr = set()
            for a1 in list_attr:
                for aa1 in a1:
                    set_attr.add(aa1)
            print('real set_attr for category:', len(set_attr))
            nominal2value[key] = {}
            nominal2value[key]['idx2value'] = list(set_attr)
            nominal2value[key]['value2idx'] = {value:idx for idx, value in enumerate(nominal2value[key]['idx2value'])}
            arr = np.asarray([[v] for v in nominal2value[key]['idx2value']])
            encoder = OneHotEncoder(sparse=False)
            nominal2value[key]['onehot_matrix'] = encoder.fit_transform(arr)
            nominal2value[key]['value2vec'] = {value:nominal2value[key]['onehot_matrix'][idx] for value, idx in nominal2value[key]['value2idx'].items()}
            print('%s is added as a nominal, whose size is %d'%(key, len(set_attr)))

        elif 0 < len(set_attr) < 300: # If one attribute only has a small number of value set, we index them
            # if key not in nominal2value:
                # nominal2value[key] = {}
            set_attr.add('') # For unseen data
            nominal2value[key] = {}
            nominal2value[key]['idx2value'] = list(set_attr)
            nominal2value[key]['value2idx'] = {value:idx for idx, value in enumerate(nominal2value[key]['idx2value'])}
            arr = np.asarray([[v] for v in nominal2value[key]['idx2value']])
            encoder = OneHotEncoder(sparse=False)
            nominal2value[key]['onehot_matrix'] = encoder.fit_transform(arr)
            nominal2value[key]['value2vec'] = {value:nominal2value[key]['onehot_matrix'][idx] for value, idx in nominal2value[key]['value2idx'].items()}
            print('%s is added as a nominal, whose size is %d'%(key, len(set_attr)))
        elif key in nominal2value:
            print('Attribute "%s" is added as a function'%(key))
        else:
            print('Attribute "%s" needs care... The size is %d.'%(key, len(set_attr)))
            print('Example value:')
            for _ in range(5):
                print(data[_][key])
            raise ValueError
        # print(nominal2value[key])
        # print(nominal2value)
        # assert False
    return attrs, nominal2value

def get_vector(d, nominal2value, attrs, has_label):
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
        # if attr in ['category', 'accessories']:
        if attr in ['category']:
            vecs = []
            for v in value.split(','):
                if v not in nominal2value[attr]['value2vec']: # This value is unseen value for that attribute
                    v = ''
                vec = nominal2value[attr]['value2vec'][v]
                vecs.append(vec)
            vec = sum(vecs)
        elif isinstance(nominal2value[attr]['value2vec'], dict):
            if value not in nominal2value[attr]['value2vec']: # This value is unseen value for that attribute
                value = ''
            vec = nominal2value[attr]['value2vec'][value]
        else:
            vec = nominal2value[attr]['value2vec'](value)
        if vec is None:
            print(attr, value)

        if isinstance(vec, list) or isinstance(vec, np.ndarray):
            vector += [*vec]
        else:
            vector += [vec]
    return vector



def build_vectors(data, nominal2value, attrs, has_label=True):
    vectors = []
    for idx, elm in enumerate(data):
        vector = get_vector(elm, nominal2value, attrs, has_label)
        vectors.append(vector)
    return np.float32(vectors)



data_train = parse('data/train.csv')

print(data_train[0].keys())

attrs, nominal2value = analyze_attribute(data_train)

data_train = build_vectors(data_train, nominal2value, attrs)
print(data_train.shape)

print()
X = data_train[:, :-1]
y = data_train[:, -1]



# Cross-Validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

other_params ={"n_estimators": 600, "max_depth": 4, "min_child_weight": 4, "colsample_bytree": 0.7}
learning_rate = 0.09
model = XGBRegressor(**other_params, learning_rate=learning_rate)
print(model)
model.fit(X_train, y_train)
# model.fit(X, y)

def write_to_csvfile(predicted, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Id', 'Predicted']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()
        # for i in range(y_pred.shape[0]):
        for i in range(predicted.shape[0]):
            writer.writerow({'Id': i, 'Predicted': '%.1f'%predicted[i]})
        csvfile.close()

# validate on the validation set
y_pred_test = model.predict(X_test)
print('Accuracy on the trainset:', np.linalg.norm(y_pred_test - y_test))
write_to_csvfile(y_pred_test, 'results/predicted_valid.csv')



# Evaluting...
data_test = parse('data/test.csv')
print(data_test[0].keys())
data_test = build_vectors(data_test, nominal2value, attrs, has_label=False)
print(data_test.shape)
y_pred = model.predict(data_test)

print('Mean of test set:', y_pred.mean())
write_to_csvfile(y_pred, 'results/submission.csv')


# 2493261.8
# 2496430.0
# 2030939.4, lr=0.5
# 1922044.0 # lr=0.1
# 2884436.8, lr=0.1, max_depth=1
# 1902594.9, lr=0.1, max_depth=2
# 1742494.8, lr=0.1, max_depth=4
# 1640406.0, lr=0.1, max_depth=3
# 1623224.2, lr=0.1, max_depth=3, min_child_weight=4
# 1620364.9, lr=0.1, max_depth=3, min_child_weight=3
# 1612684.1, lr=0.1, max_depth=3, min_child_weight=3, colsample_bytree=0.5
# 1564886.6, lr=0.1, max_depth=3, min_child_weight=3, colsample_bytree=0.7
# 1552552.4, lr=0.1, max_depth=3, min_child_weight=3, colsample_bytree=0.7, year only
# 1532948.8, lr=0.1, max_depth=4, min_child_weight=3, colsample_bytree=0.7, year only
# 1512780.6, lr=0.09, max_depth=4, min_child_weight=3, colsample_bytree=0.7, year only
# 1508084.4, lr=0.09, max_depth=4, min_child_weight=3, colsample_bytree=0.7, year only, n_estimators=600
# 1491240.9, lr=0.09, max_depth=4, min_child_weight=4, colsample_bytree=0.7, year only, n_estimators=600

# 1944438.9 # lr=0.05
# 2106857.5 # lr=0.01
# 6921284.0