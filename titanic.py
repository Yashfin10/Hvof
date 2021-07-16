'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def marks_prediction(hrs):
    X = pd.read_csv('Linear_X_Train.csv')
    Y = pd.read_csv('Linear_Y_Train.csv')

    X = X.values
    Y = Y.values

    model = LinearRegression()

    model.fit(X,Y)

    
    X_test = np.array(hrs)
    X_test = X_test.reshape((1,-1))

    return model.predict(X_test)'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

dataset = pd.read_csv('C:/Users/saima/Downloads/hiring.csv')

dataset['experience'].fillna(0, inplace=True)
dataset['test_score(out of 10)'].fillna(dataset['test_score(out of 10)'].mean(), inplace=True)
X = dataset.iloc[:, :3]

def convert_to_int(word):
    word_dict = {'one' : 1, 'two' : 2, 'three': 3, 'four' : 4, 'five' : 5, 'six' : 6, 
                 'seven' : 7, 'eight': 8, 'nine' : 9, 'ten' : 10, 'eleven' : 11, 'twelve' : 12, 'zero':0, 0:0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X, y)

pickle.dump(regressor, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[2,9,6]]))