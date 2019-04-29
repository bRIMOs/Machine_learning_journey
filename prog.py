#python3 venv /learn/bin/

import os
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import plot_model

filename = 'heart.csv'
k = 200
epochs = 4
batch_size = 100

class binary_classification_nn(object):
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(64, activation='relu', input_dim=1))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

    def train(self, data, labels, epochs, batch_size):
        self.model.fit(data, labels,
                       epochs=epochs,
                       batch_size=batch_size,)
 
    def evaluate(self, data, labels, batch_size):
        return self.model.evaluate(data, labels, batch_size=batch_size)



#x_train = np.random.random((1000, 20))
#y_train = np.random.randint(2, size=(1000, 1))
#x_test = np.random.random((100, 20))
#y_test = np.random.randint(2, size=(100, 1))

#print(x_train)
#print(y_train)

def read_from_csv(fl):
    return pd.read_csv(fl)

def show(x, y):
    sns.jointplot(x, y, kind='hex') 
    plt.tight_layout()
    plt.show()

def split_dataset(data, k):
    train = np.array([], dtype=int)
    test = np.array([], dtype=int)

    for i in range(0, k): 
        train = np.append(train, data[i])

    for i in range(k+1, len(data)):
        test = np.append(test, data[i])

    return train, test


df = read_from_csv(filename)

sex = df['sex']
chol = df['chol']

sex_train, sex_test = split_dataset(sex, k)
chol_train, chol_test = split_dataset(chol, k)

#print(df.head())

print(sex_train)


#axe1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
#axe1.set_title('Title')
#axe1.pie(sizes, shadow=True, labels=labels)
#axe1.axis('equal')


#show(sex, chol)


#print(Sequential)


nn = binary_classification_nn()
nn.train(chol_train, sex_train, epochs, batch_size)
score = nn.evaluate(chol_test, sex_test, 20)

print('Loss : %0.3f' % score[0])
print('Accuracy : %0.3f' % score[1])


#plot_self.model(self.model, to_file='self.model.png', show_shapes=True)
#os.system('display self.model.png')


