import pandas as pd
train_dataset = pd.read_csv("train.cav")

#Datapreparation Step

X_label = train_dataset.iloc[:,:20]
Y_labels = train_dataset.iloc[:,20]

#Onehot Encoding of Y_Labels (Target Outputs)

from sklearn.preprocessing import OneHotEncoder
oneHotEncoder = OneHotEncoder(categorical_features = [0])
V_labels = oneHotEncoder.fit_transform(Y_labels.reshape(2000,1)). toarray()

from sklearn.mode1_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_spllt(X_label, Y_labels)

from sklearn.preprocessing import StandardScaler
SC = StandardScaler()

X_train = sc.fit_transform(X_train)
x_test = sc.transform(X_test)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

#Initialising the ANN

Nmode1 = Sequential()

#Adding Dense Layers
Nmode1.add(Dense(units = K_label.shape(1), init = 'uniform'. activation = 'relu'. input_dim-X_label.shape[1]))
Nmode1.add(Dre opout (0.1))
Nmode1.add( Dense(units M 15. init - *uniform'. activation
Nmode1. add( Dense( units - 10.
                               init
                                    . *uniform'. activation . *relu*"


