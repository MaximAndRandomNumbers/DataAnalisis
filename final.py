import numpy as np

np.random.seed(2)
import tensorflow as tf

tf.random.set_seed(2)
import pandas as pd
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

warnings.filterwarnings("ignore")
np.set_printoptions(precision=3)
pd.set_option('precision', 3)

data = pd.read_csv('train.csv', index_col=0)
data_test = pd.read_csv('test.csv', index_col=0)

columns_to_drop = ['Internet', 'TotalCharges', 'Contract', 'Gender']
data.drop(columns_to_drop, axis=1, inplace=True)

data = data.replace({
    'Married': {'Yes': 1, 'No': 0},
    'Phone': {'Yes': 1, 'No': 0},
    'MultiplePhones': {'Yes': 1, 'No': 0, 'No phone service': 2},
    'Security': {'Yes': 1, 'No': 0, 'No internet service': 2},
    'Backup': {'Yes': 1, 'No': 0, 'No internet service': 2},
    'Insurance': {'Yes': 1, 'No': 0, 'No internet service': 2},
    'Support': {'Yes': 1, 'No': 0, 'No internet service': 2},
    'TV': {'Yes': 1, 'No': 0, 'No internet service': 2},
    'Movies': {'Yes': 1, 'No': 0, 'No internet service': 2},
    'EBilling': {'Yes': 1, 'No': 0},
    'PaymentMethod': {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2,
                      'Credit card (automatic)': 3},
    'Dependents': {'Yes': 1, 'No': 0}
})

data_test.drop(columns_to_drop, axis=1, inplace=True)
data_test = data_test.replace({
    'Married': {'Yes': 1, 'No': 0},
    'Phone': {'Yes': 1, 'No': 0},
    'MultiplePhones': {'Yes': 1, 'No': 0, 'No phone service': 2},
    'Security': {'Yes': 1, 'No': 0, 'No internet service': 2},
    'Backup': {'Yes': 1, 'No': 0, 'No internet service': 2},
    'Insurance': {'Yes': 1, 'No': 0, 'No internet service': 2},
    'Support': {'Yes': 1, 'No': 0, 'No internet service': 2},
    'TV': {'Yes': 1, 'No': 0, 'No internet service': 2},
    'Movies': {'Yes': 1, 'No': 0, 'No internet service': 2},
    'EBilling': {'Yes': 1, 'No': 0},
    'PaymentMethod': {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2,
                        'Credit card (automatic)': 3},
    'Dependents': {'Yes': 1, 'No': 0}
})

np_data_train = data.values.astype(np.float32)
np_data_test = data_test.values.astype(np.float32)

X_neuro_train = np_data_train[:, 0:data.shape[1] - 1]
Y_neuro_train = np_data_train[:, data.shape[1] - 1]

X_neuro_test = np_data_test

model = Sequential()
model.add(Dense(100, input_dim=data.shape[1] - 1, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(X_neuro_train, Y_neuro_train, epochs=36, batch_size=1)
scoresTrain = model.evaluate(X_neuro_train, Y_neuro_train)

print(scoresTrain[1])
predictions = model.predict(X_neuro_test)

for i in range(len(predictions)):
    if predictions[i] < 0.5:
        predictions[i] = False
    else:
        predictions[i] = True

temp = pd.read_csv('sample_submission.csv', index_col=0)
for i in range(len(temp['Churn'])):
    temp['Churn'][i] = predictions[i]

temp.to_csv('FINAL_RES.csv')
