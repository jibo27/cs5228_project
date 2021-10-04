import os
import csv
from numpy.core.numeric import normalize_axis_tuple
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import numpy as np
from datetime import datetime
import pickle

from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score

np.random.seed(0)


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
        value = 0
    else:
        value = datetime.strptime(s, '%d-%b-%Y').toordinal()
    return value 

attr_ignored = ['listing_id', 'title', 'description', 'features', 'accessories',
                'model']

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
            if key in ['original_reg_date', 'reg_date', 'lifespan' ]:
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
        if 0 < len(set_attr) < 300: # If one attribute only has a small number of value set, we index them
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
        if isinstance(nominal2value[attr]['value2vec'], dict):
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
data_test = parse('data/test.csv')

print(data_train[0].keys())
print(data_test[0].keys())

attrs, nominal2value = analyze_attribute(data_train)

data_train = build_vectors(data_train, nominal2value, attrs)
data_test = build_vectors(data_test, nominal2value, attrs, has_label=False)
print(data_train.shape)
print(data_test.shape)

print()
X_train = data_train[:, :-1]
y_train = data_train[:, -1]
model = XGBRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(data_test)

def write_to_csvfile(predicted, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Id', 'Predicted']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()
        for i in range(y_pred.shape[0]):
            writer.writerow({'Id': i, 'Predicted': '%.1f'%predicted[i]})
        csvfile.close()

write_to_csvfile(y_pred, 'submission.csv')

# validate on the training set
y_pred_train = model.predict(X_train)
print('Accuracy on the trainset:', np.linalg.norm(y_pred_train - y_train))
write_to_csvfile(y_pred_train, 'predicted_train.csv')