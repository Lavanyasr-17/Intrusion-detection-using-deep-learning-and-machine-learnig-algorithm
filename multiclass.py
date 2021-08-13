#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
import seaborn as sns
from tensorflow.keras import regularizers

from tensorflow.keras import layers
from tensorflow.keras import losses

from collections import Counter


import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer


# In[3]:


data= pd.read_csv("UNSW_NB15_test-set.csv")
print('-------Data--------')
print(data['attack_cat'].value_counts())
print(len(data))
print('-------------------------')
data


# In[4]:


#data.drop(data[data['attack_cat']=='Generic'].index, inplace = True)
#data.drop(data[data['attack_cat']=='Fuzzers'].index, inplace = True)
#data.drop(data[data['attack_cat']=='DoS'].index, inplace = True)
#data.drop(data[data['attack_cat']=='Backdoor'].index, inplace = True)
#data.drop(data[data['attack_cat']=='Shellcode'].index, inplace = True)
#data.drop(data[data['attack_cat']=='Worms'].index, inplace = True)

#print(data['attack_cat'].value_counts())


# In[5]:


#print(data['attack_cat'].value_counts())


# In[6]:


data.isnull().sum()


# In[7]:


data.info()


# In[8]:


print(data['service'].value_counts())
print(data['proto'].value_counts())
print(data['state'].value_counts())


# In[9]:


data=data.drop(['service'],axis=1)
data=data.drop(['id'],axis=1)


# In[10]:


from sklearn import preprocessing
le=preprocessing.LabelEncoder()
data['proto'] = le.fit_transform(data['proto'])
data['state'] = le.fit_transform(data['state'])
data.replace({'attack_cat':{'Normal':0,'Generic':1,'Exploits':2,'Fuzzers':3,'DoS':4,'Reconnaissance':5,'Analysis':6,'Backdoor':7,'Shellcode':8,'Worms':9,}},inplace=True)
print(data['attack_cat'].value_counts())


# In[11]:


data = data.apply(pd.to_numeric)


# In[12]:


data.info()


# In[13]:


data.round(decimals=2)
data = data.astype(int)
data


# In[14]:


data.info()


# In[15]:





# In[16]:


data=data.drop(['sload'],axis=1)
data=data.drop(['dload'],axis=1)


# In[17]:


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr
 
corr_features = correlation(data, 0.8)
print('correlated features: ', set(corr_features) )


# In[18]:


data.drop(labels=corr_features, axis=1, inplace=True)


# In[19]:


features = data.drop(['attack_cat'],axis=1)
target = data['attack_cat']
X_Train,X_Test,Y_Train,Y_Test=train_test_split(features,target,test_size = 0.2,random_state=42)


# In[20]:


X = X_Train
Y = Y_Train
C = Y_Test
T = X_Test
print(Y)
print(X)


# In[21]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# transform data
trainX = scaler.fit_transform(X)
# summarize transformed data
np.set_printoptions(precision=3)
#print(trainX[0:5,:])


# In[22]:


scaler = StandardScaler()
# transform data
testT = scaler.fit_transform(T)
# summarize transformed data
np.set_printoptions(precision=3)
#print(testT[0:5,:])


# In[23]:


from keras.utils.np_utils import to_categorical
y_train1 = np.array(Y)
y_test1 = np.array(C)
y_train= to_categorical(y_train1)
y_test= to_categorical(y_test1)


# In[24]:


#cnn
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding
from keras.layers import Convolution1D,MaxPooling1D, Flatten
from keras.datasets import imdb
from keras import backend as K
import pandas as pd
from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import Normalizer
from keras.models import Sequential
from keras.layers import Convolution1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.utils import np_utils
import numpy as np
import h5py
from keras import callbacks
from keras.layers import LSTM, GRU, SimpleRNN
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.layers.convolutional import MaxPooling1D


# In[25]:


lstm_output_size = 128

cnn = Sequential()
cnn.add(Convolution1D(64, 2, padding="same",activation="relu",input_shape=(26,1)))
cnn.add(MaxPooling1D(pool_size=(2)))
cnn.add(Flatten())
cnn.add(Dense(128, activation="relu"))
cnn.add(Dropout(0.1))
cnn.add(Dense(10, activation="softmax"))
print(cnn.summary())


# In[26]:


X_train = np.reshape(trainX, (trainX.shape[0], trainX.shape[1],1))
X_test = np.reshape(testT, (testT.shape[0], testT.shape[1],1))
cnn.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])
cnn.fit(X_train, y_train,epochs=50)


# In[27]:


loss, accuracy = cnn.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))


# In[28]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
y_pred1 = cnn.predict_classes(X_test)
y_pred= to_categorical(y_pred1)
np.savetxt('expected.txt', y_test, fmt='%01d')
np.savetxt('predicted.txt', y_pred, fmt='%01d')
cnn_accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred , average="weighted")
precision = precision_score(y_test, y_pred , average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("confusion matrix")
print("----------------------------------------------")
print("accuracy")
print("%.6f" %cnn_accuracy)
print("racall")
print("%.6f" %recall)
print("precision")
print("%.6f" %precision)
print("f1score")
print("%.6f" %f1)
cm = confusion_matrix(y_test1, y_pred1)
print(cm)
print("==============================================")


# In[29]:


from sklearn.metrics import classification_report
print(classification_report(y_test1, y_pred1, target_names=['Normal','Generics','Exploits','Fuzzers','DoS','Reconnaissance','Analysis','Backdoor','Shellcode','Worms']))


# # RNN

# In[30]:



from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical


# In[31]:


rnn = Sequential()
rnn.add(SimpleRNN(64,input_dim=27, return_sequences=True))  
rnn.add(Dropout(0.1))
rnn.add(SimpleRNN(64,input_dim=27, return_sequences=False))  
rnn.add(Dropout(0.1))
rnn.add(Dense(10))
rnn.add(Activation('softmax'))
print(rnn.summary())


# In[33]:


X_train = np.reshape(trainX, (trainX.shape[0],1, trainX.shape[1]))
X_test = np.reshape(testT, (testT.shape[0],1, testT.shape[1]))
rnn.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])
rnn.fit(X_train, y_train,epochs=50)


# In[35]:


loss, accuracy = rnn.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))


# In[36]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
y_pred1 = rnn.predict_classes(X_test)
y_pred= to_categorical(y_pred1)
np.savetxt('expected.txt', y_test, fmt='%01d')
np.savetxt('predicted.txt', y_pred, fmt='%01d')
rnn_accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred , average="weighted")
precision = precision_score(y_test, y_pred , average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("confusion matrix")
print("----------------------------------------------")
print("accuracy")
print("%.6f" %rnn_accuracy)
print("racall")
print("%.6f" %recall)
print("precision")
print("%.6f" %precision)
print("f1score")
print("%.6f" %f1)
cm = confusion_matrix(y_test1, y_pred1)
print(cm)
print("==============================================")


# In[37]:


from sklearn.metrics import classification_report
print(classification_report(y_test1, y_pred1, target_names=['Normal','Generics','Exploits','Fuzzers','DoS','Reconnaissance','Analysis','Backdoor','Shellcode','Worms']))


# # NB

# In[39]:


from sklearn.naive_bayes import GaussianNB
model= GaussianNB()
model.fit(X_Train, Y_Train)
prediction= model.predict(X_Test) #prediction based on testing_set
a3 = accuracy_score(Y_Test, prediction)
r3 = recall_score(Y_Test, prediction , average="weighted")
p3 = precision_score(Y_Test, prediction , average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")
print("confusion matrix")
print("----------------------------------------------")
print("accuracy")
print("%.3f" %a3)
print("racall")
print("%.3f" %r3)
print("precision")
print("%.3f" %p3)
print("f1 score")
print("%.3f" %f1)
expected= Y_Train #expected result stored in training_set
predicted= model.predict(X_Train)
print('Classification report:')
print(classification_report(expected,predicted,target_names=['Normal','Generics','Exploits','Fuzzers','DoS','Reconnaissance','Analysis','Backdoor','Shellcode','Worms'])) #calculating classification report
print('Confusion matrix:')
print(confusion_matrix(expected,predicted)) #calculating confusion matrix


# # SVM

# In[40]:


from sklearn.svm import SVC
model = SVC(kernel='sigmoid',C=0.1,random_state = 1)
model.fit(trainX, y_train1)


# In[41]:


predict = model.predict(testT)


# In[43]:


from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
svm_accuracy=metrics.accuracy_score(y_test1, predict)
print("Accuracy:",svm_accuracy)
r4 = recall_score(y_test1, predict , average="weighted")
p4 = precision_score(y_test1, predict , average="weighted")
f1 = f1_score(y_test1, predict, average="weighted")


print("racall")
print("%.3f" %r4)
print("precision")
print("%.3f" %p4)
print("f1 score")
print("%.3f" %f1)
print(classification_report(y_test1, predict,target_names=['Normal','Generics','Exploits','Fuzzers','DoS','Reconnaissance','Analysis','Backdoor','Shellcode','Worms']))
print("confusion matrix")
print("----------------------------------------------")
cm = confusion_matrix(y_test1, predict)
print(cm)


# In[44]:


#Dt
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(trainX, y_train1)


# In[45]:


predict = dtc.predict(testT)


# In[46]:


from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
dtc_accuracy=metrics.accuracy_score(y_test1, predict)
print("Accuracy:",dtc_accuracy)
print(classification_report(y_test1, predict,target_names=['Normal','Generics','Exploits','Fuzzers','DoS','Reconnaissance','Analysis','Backdoor','Shellcode','Worms']))
cm = confusion_matrix(y_test1, predict)
print(cm)


# In[50]:


report = pd.DataFrame(index=[1,2,3,4,5])
report['Algorithm'] = ['Convolutional Neural Network(CNN)', 'Recrrent Neural network(RNN)','Naive Bayes(NB)','Support vector Machine(SVM)', 'Decision Tree(DT)']
report['Accuracy'] = [cnn_accuracy*100, rnn_accuracy*100,a3*100,svm_accuracy*100,dtc_accuracy*100]
report


# In[51]:


fig, ax = plt.subplots(figsize=(10,5))
sns.lineplot(x="Algorithm", y="Accuracy", data=report, ax=ax)
plt.show()


# In[ ]:




