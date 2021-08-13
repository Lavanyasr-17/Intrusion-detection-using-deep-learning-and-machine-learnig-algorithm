#!/usr/bin/env python
# coding: utf-8

# In[132]:


import pandas as pd
import seaborn as sns
col_names = ["id"
                 ,"dur","proto"
                 ,"service","state","spkts","dpkts","sbytes","dbytes","rate","sttl","dttl","sload","dload","sloss"
                 ,"dloss","sinpkt","dinpkt","sjit","djit","swin","stcpb","dtcpb","dwin","tcprtt","synack"
                 ,"ackdat","smean","dmean","trans_depth","response_body_len","ct_srv_src","ct_state_ttl","ct_dst_ltm","ct_src_dport_ltm",
                 "ct_dst_sport_ltm","ct_dst_src_ltm","is_ftp_login","ct_ftp_cmd","ct_flw_http_mthd","ct_src_ltm","ct_srv_dst","is_sm_ips_ports","attack_cat","label"]
train_data= pd.read_csv("UNSW_NB15_training-set.csv",header=None,names=col_names)
col_names = ["id"
                 ,"dur","proto"
                 ,"service","state","spkts","dpkts","sbytes","dbytes","rate","sttl","dttl","sload","dload","sloss"
                 ,"dloss","sinpkt","dinpkt","sjit","djit","swin","stcpb","dtcpb","dwin","tcprtt","synack"
                 ,"ackdat","smean","dmean","trans_depth","response_body_len","ct_srv_src","ct_state_ttl","ct_dst_ltm","ct_src_dport_ltm",
                 "ct_dst_sport_ltm","ct_dst_src_ltm","is_ftp_login","ct_ftp_cmd","ct_flw_http_mthd","ct_src_ltm","ct_srv_dst","is_sm_ips_ports","attack_cat","label"]
test_data = pd.read_csv('UNSW_NB15_testing-set.csv',header=None, names = col_names)
    #represent data in form of bar graph
    
     #distribution of attacks and normal traffic
f, axes = plt.subplots(2, 2, figsize=(12, 12))
f.suptitle("---------------------------------\nTrain data length 82332\n---------------------------------\nTest data length 175341\n---------------------------------\n")
    # Create the plots using seaborn package
sns.countplot(x="label", data=train_data, ax=axes[0,0])
sns.countplot(x="label", data=test_data, ax=axes[0,1])
sns.countplot(x="attack_cat", data=train_data, ax=axes[1,0], order = train_data['attack_cat'].value_counts().index)
sns.countplot(x="attack_cat", data=test_data, ax=axes[1,1], order = test_data['attack_cat'].value_counts().index)

    # plot titles
axes[0,0].set_title("Training data distribution")
axes[1,0].set_title("Training data distribution")
axes[0,1].set_title("Testing data distribution")
axes[1,1].set_title("Testing data distribution")

    # Rotate xticks for readability
axes[1,0].tick_params('x', labelrotation=45)
axes[1,1].tick_params('x', labelrotation=45)

    # Change the xtick labels for attack / normal
axes[0,0].set_xticklabels(["Normal", "Attack"])
axes[0,1].set_xticklabels(["Normal", "Attack"])

    # Adding some space between the plots for y labels
plt.subplots_adjust(wspace=0.5)
    


# 
# # RNN

# In[46]:


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



# In[47]:


#uploading training dataset
test_data= pd.read_csv("UNSW_NB15_testing-set.csv",header=None)
print('-------Test data--------')
print(test_data[43].value_counts())
print(test_data.shape)
print('-------------------------')
test_data.drop(test_data.index[0],inplace = True)


# In[ ]:





# In[48]:


#uploading training dataset
train_data= pd.read_csv("UNSW_NB15_training-set.csv",header= None)
print('-------Train data--------')
print(train_data[43].value_counts())
print(len(train_data))
print('-------------------------')
train_data.drop(train_data.index[0],inplace = True)
train_data


# In[ ]:





# In[49]:


#preprocessing
#Evaluation of the training dataset
train_data.isnull().sum()


# In[50]:



train_data.info()


# In[51]:


train_data=train_data.drop([2,3,43],axis=1)
train_data=train_data.drop([0],axis=1)
test_data=test_data.drop([2,3,43],axis=1)
test_data=test_data.drop([0],axis=1)


# In[52]:


from sklearn import preprocessing
le=preprocessing.LabelEncoder()
'''train_data[43] = le.fit_transform(train_data[43])
test_data[43] = le.fit_transform(test_data[43])'''
train_data[4] = le.fit_transform(train_data[4])
test_data[4] = le.fit_transform(test_data[4])


# In[53]:


train_data = train_data.apply(pd.to_numeric)
test_data = test_data.apply(pd.to_numeric)


# In[54]:


train_data.info()


# In[55]:


'''list=[1,9,12,13,16,17,18,19,24,25,26]
for x in list:
  '''
train_data.round(decimals=2)
train_data = train_data.astype(int)
test_data.round(decimals=2)
test_data = test_data.astype(int)
train_data


# In[56]:


train_data.info()


# In[57]:


features = train_data.iloc[:,0:40]
target = train_data.iloc[:,40]
X_Train,X_Test,Y_Train,Y_Test=train_test_split(features,target,test_size = 0.2,random_state=42)


# In[58]:


X = X_Train
Y = Y_Train
C = Y_Test
T = X_Test
print(Y)
print(X)


# In[59]:


from __future__ import print_function
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from tensorflow.keras.preprocessing import sequence
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding
from tensorflow.keras.layers import LSTM, SimpleRNN, GRU
from tensorflow.keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger


# In[60]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# transform data
trainX = scaler.fit_transform(X)
# summarize transformed data
np.set_printoptions(precision=3)
#print(trainX[0:5,:])


# In[61]:


scaler = StandardScaler()
# transform data
testT = scaler.fit_transform(T)
# summarize transformed data
np.set_printoptions(precision=3)
#print(testT[0:5,:])


# In[62]:


y_train = np.array(Y)
y_test = np.array(C)


# In[63]:



X_train = np.reshape(trainX, (trainX.shape[0],1, trainX.shape[1]))
X_test = np.reshape(testT, (testT.shape[0],1, testT.shape[1]))


# In[64]:




batch_size = 64

# 1. define the network
model = Sequential()
model.add(SimpleRNN(8,input_dim=40, return_sequences=True)) 
model.add(Dropout(0.1))
model.add(SimpleRNN(8, return_sequences=False)) 
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[65]:


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=batch_size, epochs=50)


# In[66]:


loss, accuracy = model.evaluate(X_test, y_test)
y_pred = model.predict_classes(X_test)


# In[67]:


y_train1 = y_test
y_pred = model.predict_classes(X_test)
a1 = accuracy_score(y_train1, y_pred)
r1 = recall_score(y_train1, y_pred , average="binary")
p1 = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")


# In[68]:


print("confusion matrix")
print("----------------------------------------------")
print("accuracy")
print("%.3f" %a1)
print("racall")
print("%.3f" %r1)
print("precision")
print("%.3f" %p1)
print("f1 score")
print("%.3f" %f1)
from sklearn.metrics import classification_report
print(classification_report(y_train1, y_pred))
print("==============================================")


# # CNN

# In[69]:


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



# In[70]:


#uploading training dataset

test_data= pd.read_csv("UNSW_NB15_testing-set.csv",header=None)
print('-------Test data--------')
print(test_data[43].value_counts())
print(test_data.shape)
print('-------------------------')
test_data.drop(test_data.index[0],inplace = True)
test_data


# In[ ]:





# In[71]:


#uploading training dataset

train_data= pd.read_csv("UNSW_NB15_training-set.csv",header= None)
print('-------Train data--------')
print(train_data[43].value_counts())
print(len(train_data))
print('-------------------------')
train_data.drop(train_data.index[0],inplace = True)
train_data


# In[72]:


#preprocessing
#Evaluation of the training dataset
train_data.isnull().sum()


# In[73]:



train_data.info()


# In[74]:


train_data=train_data.drop([2,3,43],axis=1)
train_data=train_data.drop([0],axis=1)
test_data=test_data.drop([2,3,43],axis=1)
test_data=test_data.drop([0],axis=1)


# In[75]:


from sklearn import preprocessing
le=preprocessing.LabelEncoder()
'''train_data[43] = le.fit_transform(train_data[43])
test_data[43] = le.fit_transform(test_data[43])'''
train_data[4] = le.fit_transform(train_data[4])
test_data[4] = le.fit_transform(test_data[4])


# In[76]:


train_data = train_data.apply(pd.to_numeric)
test_data = test_data.apply(pd.to_numeric)


# In[77]:


train_data.info()


# In[78]:


'''list=[1,9,12,13,16,17,18,19,24,25,26]
for x in list:
  '''
train_data.round(decimals=2)
train_data = train_data.astype(int)
test_data.round(decimals=2)
test_data = test_data.astype(int)
train_data


# In[79]:


train_data.info()


# In[80]:


X = train_data.iloc[:,0:40]
Y = train_data.iloc[:,40]
C = test_data.iloc[:,40]
T = test_data.iloc[:,0:40]
print(Y)
print(X)


# In[81]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# transform data
trainX = scaler.fit_transform(X)
# summarize transformed data
np.set_printoptions(precision=3)
#print(trainX[0:5,:])


# In[82]:


scaler = StandardScaler()
# transform data
testT = scaler.fit_transform(T)
# summarize transformed data
np.set_printoptions(precision=3)
#print(testT[0:5,:])


# In[83]:


y_train = np.array(Y)
y_test = np.array(C)


# In[84]:



X_train = np.reshape(trainX, (trainX.shape[0], trainX.shape[1],1))
X_test = np.reshape(testT, (testT.shape[0], testT.shape[1],1))


# In[85]:


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


# In[86]:


lstm_output_size = 128

cnn = Sequential()
cnn.add(Convolution1D(64, 3, padding="same",activation="relu",input_shape=(40,1)))
cnn.add(MaxPooling1D(pool_size=(2)))
cnn.add(Flatten())
cnn.add(Dense(128, activation="relu"))
cnn.add(Dropout(0.1))
cnn.add(Dense(1, activation="sigmoid"))
print(cnn.summary())


# In[87]:


cnn.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])
cnn.fit(X_train, y_train,epochs=50)


# In[88]:


loss, accuracy = cnn.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))


# In[89]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
y_pred = cnn.predict_classes(X_test)
#np.savetxt('/content/drive/MyDrive/predicted1.txt', y_pred, fmt='%01d')
a2 = accuracy_score(y_test, y_pred)
r2 = recall_score(y_test, y_pred , average="binary")
p2 = precision_score(y_test, y_pred , average="binary")
f1 = f1_score(y_test, y_pred, average="binary")

print("confusion matrix")
print("----------------------------------------------")
print("accuracy")
print("%.6f" %a2)
print("racall")
print("%.6f" %r2)
print("precision")
print("%.6f" %p2)
print("f1score")
print("%.6f" %f1)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("==============================================")


# In[180]:





# # NB

# In[90]:


import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
import seaborn as sns
from tensorflow.keras import regularizers
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras import layers
from tensorflow.keras import losses

from collections import Counter
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score,mean_squared_error,mean_absolute_error)

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
from sklearn.naive_bayes import GaussianNB


# In[91]:


#uploading training dataset
test_data= pd.read_csv("UNSW_NB15_testing-set.csv",header=None)
print('-------Test data--------')
print(test_data[43].value_counts())
print(test_data.shape)
print('-------------------------')
test_data.drop(test_data.index[0],inplace = True)
test_data


# In[ ]:





# In[92]:


train_data= pd.read_csv("UNSW_NB15_training-set.csv",header= None)
print('-------Train data--------')
print(train_data[43].value_counts())
print(len(train_data))
print('-------------------------')
train_data.drop(train_data.index[0],inplace = True)
train_data


# In[ ]:





# In[93]:


train_data.isnull().sum()
train_data.info()


# In[94]:


train_data=train_data.drop([2,3,43],axis=1)
train_data=train_data.drop([0],axis=1)
test_data=test_data.drop([2,3,43],axis=1)
test_data=test_data.drop([0],axis=1)


# In[95]:


from sklearn import preprocessing
le=preprocessing.LabelEncoder()
'''train_data[43] = le.fit_transform(train_data[43])
test_data[43] = le.fit_transform(test_data[43])'''
train_data[4] = le.fit_transform(train_data[4])
test_data[4] = le.fit_transform(test_data[4])


# In[96]:


train_data = train_data.apply(pd.to_numeric)
test_data = test_data.apply(pd.to_numeric)


# In[97]:


train_data.info()


# In[98]:


train_data.round(decimals=2)
train_data = train_data.astype(int)
test_data.round(decimals=2)
test_data = test_data.astype(int)
train_data


# In[99]:


train_data.info()


# In[100]:


features = train_data.iloc[:,0:40]
target = train_data.iloc[:,40]
X_Train,X_Test,Y_Train,Y_Test=train_test_split(features,target,test_size = 0.2,random_state=2)


# In[101]:


model= GaussianNB()
model.fit(X_Train, Y_Train)
prediction= model.predict(X_Test) #prediction based on testing_set
a3 = accuracy_score(Y_Test, prediction)
r3 = recall_score(Y_Test, prediction , average="binary")
p3 = precision_score(Y_Test, prediction , average="binary")
f1 = f1_score(y_test, y_pred, average="binary")
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
print(classification_report(expected,predicted)) #calculating classification report
print('Confusion matrix:')
print(confusion_matrix(expected,predicted)) #calculating confusion matrix


# # SVM

# In[102]:


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


# In[103]:


#uploading training dataset
test_data= pd.read_csv("UNSW_NB15_testing-set.csv",header=None)
print('-------Test data--------')
print(test_data[43].value_counts())
print(test_data.shape)
print('-------------------------')
test_data.drop(test_data.index[0],inplace = True)
test_data


# In[ ]:





# In[104]:


#uploading training dataset
train_data= pd.read_csv("UNSW_NB15_training-set.csv",header= None)
print('-------Train data--------')
print(train_data[43].value_counts())
print(len(train_data))
print('-------------------------')
train_data.drop(train_data.index[0],inplace = True)
train_data


# In[ ]:





# In[105]:


#preprocessing
#Evaluation of the training dataset
train_data.isnull().sum()


# In[106]:



train_data.info()


# In[107]:


train_data=train_data.drop([2,3,43],axis=1)
train_data=train_data.drop([0],axis=1)
test_data=test_data.drop([2,3,43],axis=1)
test_data=test_data.drop([0],axis=1)


# In[108]:


from sklearn import preprocessing
le=preprocessing.LabelEncoder()
'''train_data[43] = le.fit_transform(train_data[43])
test_data[43] = le.fit_transform(test_data[43])'''
train_data[4] = le.fit_transform(train_data[4])
test_data[4] = le.fit_transform(test_data[4])


# In[109]:


train_data = train_data.apply(pd.to_numeric)
test_data = test_data.apply(pd.to_numeric)


# In[110]:


train_data.info()


# In[111]:


'''list=[1,9,12,13,16,17,18,19,24,25,26]
for x in list:
  '''
train_data.round(decimals=2)
train_data = train_data.astype(int)
test_data.round(decimals=2)
test_data = test_data.astype(int)
train_data


# In[112]:


train_data.info()


# In[113]:



features = train_data.iloc[:,0:40]
target = train_data.iloc[:,40]
X_Train,X_Test,Y_Train,Y_Test=train_test_split(features,target,test_size = 0.2,random_state=2)


# In[114]:


'''X = X_Train
Y = Y_Train
C = Y_Test
T = X_Test
print(Y)
print(X)'''


# In[115]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# transform data
trainX = scaler.fit_transform(X_Train)


# In[116]:


scaler = StandardScaler()
# transform data
testT = scaler.fit_transform(X_Test)


# In[117]:


y_train = np.array(Y_Train)
y_test = np.array(Y_Test)


# In[118]:



'''X_train = np.reshape(trainX, (trainX.shape[0], trainX.shape[1],1))
X_test = np.reshape(testT, (testT.shape[0], testT.shape[1],1))'''


# In[119]:


from sklearn.svm import SVC
model = SVC(kernel='sigmoid',C=0.1,random_state = 1)
model.fit(trainX, y_train)


# In[120]:


predict = model.predict(testT)


# In[124]:


from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
a4=metrics.accuracy_score(Y_Test, predict)
print("Accuracy:",a4)
a4 = accuracy_score(Y_Test, predict)
r4 = recall_score(Y_Test, predict , average="binary")
p4 = precision_score(Y_Test, predict , average="binary")
f1 = f1_score(Y_Test, predict, average="binary")
print("confusion matrix")
print("----------------------------------------------")
print("accuracy")
print("%.3f" %a4)
print("racall")
print("%.3f" %r4)
print("precision")
print("%.3f" %p4)
print("f1 score")
print("%.3f" %f1)
print(classification_report(Y_Test, predict))


# In[126]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(trainX, y_train)


# In[127]:


predict = dtc.predict(testT)


# In[129]:


from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
dtc_accuracy=metrics.accuracy_score(Y_Test, predict)
print("Accuracy:",dtc_accuracy)
print(classification_report(Y_Test, predict))
cm = confusion_matrix(Y_Test, predict)
print(cm)


# In[130]:


report = pd.DataFrame(index=[1,2,3,4,5])
report['Algorithm'] = ['Convolutional Neural Network(CNN)', 'Recrrent Neural network(RNN)','Naive Bayes(NB)','Support vector Machine(SVM)', 'Decision Tree(DT)']
report['Accuracy'] = [a1*100, a2*100,a3*100,a4*100,dtc_accuracy*100]
report


# In[131]:


fig, ax = plt.subplots(figsize=(10,5))
sns.lineplot(x="Algorithm", y="Accuracy", data=report, ax=ax)
plt.show()


# In[92]:






def main(file):
    label = Label(root, bg='white',text='RESULT', font = ('Helvetica', 12, "bold"), fg="red")
    label.pack()
    button_explore = Button(root,text = "Analysis of dataset",bg='white', padx=20,command = analysis).place(anchor=W,x=100, y=300)
    button_explore = Button(root,text = "Report",bg='white', padx=20,command = order).place(anchor=W,x=100, y=400)
    
   
    
    
    import numpy as np 
    import matplotlib.pyplot as plt 
    import matplotlib
    matplotlib.use('TkAgg')
    import numpy as np
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    label1 = Label(root,bg='white', text='Accuracy Metrics of RNN:')
    label1.pack()
    label = Label(root,bg='white', text='=============================================')
    label.pack()
    a=[]
    label2 = Label(root, bg='white',text='')
    label3 = Label(root, bg='white',text='')
    label4 = Label(root, bg='white',text='')
    label5 = Label(root, bg='white',text='')
    label2.config(text='Accuracy: '+str(a1))
    label2.pack()
    label3.config(text='Recall: '+str(r1))
    label3.pack()
    label4.config(text='Precision:'+str(p1))
    label4.pack()
    label5 = Label(root,bg='white', text='=============================================')
    label5.pack()
    a.append(a1)
    a.append(r1)
    a.append(p1)
    label1 = Label(root,bg='white', text='Accuracy Metrics of CNN:')
    label1.pack()
    label = Label(root,bg='white', text='=============================================')
    label.pack()
    b=[]
    label2 = Label(root, bg='white',text='')
    label3 = Label(root, bg='white',text='')
    label4 = Label(root, bg='white',text='')
    label5 = Label(root, bg='white',text='')
    label2.config(text='Accuracy: '+str(a2))
    label2.pack()
    label3.config(text='Recall: '+str(r2))
    label3.pack()
    label4.config(text='Precision:'+str(p2))
    label4.pack()
    label5 = Label(root,bg='white', text='=============================================')
    label5.pack()
    b.append(a2)
    b.append(r2)
    b.append(p2)
    label1 = Label(root,bg='white', text='Accuracy Metrics of Naive Bayes:')
    label1.pack()
    label = Label(root,bg='white', text='=============================================')
    label.pack()
    c=[]
    label2 = Label(root, bg='white',text='')
    label3 = Label(root, bg='white',text='')
    label4 = Label(root, bg='white',text='')
    label5 = Label(root, bg='white',text='')
    label2.config(text='Accuracy: '+str(a3))
    label2.pack()
    label3.config(text='Recall: '+str(r3))
    label3.pack()
    label4.config(text='Precision:'+str(p3))
    label4.pack()
    label5 = Label(root,bg='white', text='=============================================')
    label5.pack()
    c.append(a3)
    c.append(r3)
    c.append(p3)
    label1 = Label(root,bg='white', text='Accuracy Metrics of SVM:')
    label1.pack()
    label = Label(root,bg='white', text='=============================================')
    label.pack()
    d=[]
    label2 = Label(root, bg='white',text='')
    label3 = Label(root, bg='white',text='')
    label4 = Label(root, bg='white',text='')
    label5 = Label(root, bg='white',text='')
    label2.config(text='Accuracy: '+str(a4))
    label2.pack()
    label3.config(text='Recall: '+str(r4))
    label3.pack()
    label4.config(text='Precision:'+str(p4))
    label4.pack()
    label5 = Label(root,bg='white', text='=============================================')
    label5.pack()
    d.append(a4)
    d.append(r4)
    d.append(p4)
    
    label = Label(root, bg='white',text='Graphical Representation',font = ('Helvetica', 10, "bold"), fg="black")
    label.pack()
    
    
    n=3
    r = np.arange(n)

    
    Y = a
    Z = b
    zq=c
    q=d
    width=0.25
    
    
    fig = Figure(figsize=(6,10))
    
    plt=fig.add_subplot()
    
    plt.bar(r, Y, width=0.2,label = 'RNN')
    plt.bar(r+0.2, Z,width=0.2, label = 'CNN')
    plt.bar(r+0.4, zq,width=0.2, label = 'Naive Bayes')
    plt.bar(r+0.6, q,width=0.2, label = 'SVM')
    
    plt.legend()
    r=zq[0]
    c=q[0]
    s=Y[0]
    n=Z[0]
    
   
    
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()
    canvas.draw()
    
   

    
    


# In[93]:


def browseFiles():
    filename = filedialog.askopenfilename(initialdir = "/downloads",
                                          title = "Select a File",
                                          filetypes = (("CSV files",
                                                        "*.csv*"),
                                                       ("all files",
                                                        "*.*")))
    delete()
    main(filename)
def delete():
    instruction1.pack_forget()
    instructio1.pack_forget()
    instructio1.pack_forget()
    instruction2.pack_forget()

def B():
    label1 = Label(root,bg='white', text='Accuracy Metrics of CNN:')
    label1.pack()
    label = Label(root,bg='white', text='=============================================')
    label.pack()
    l=[]
    label2 = Label(root, bg='white',text='')
    label3 = Label(root, bg='white',text='')
    label4 = Label(root, bg='white',text='')
    label5 = Label(root, bg='white',text='')
    a= 0.944804
    r=0.944804
    p=0.947369
    label2.config(text='Accuracy: '+str(a))
    label2.pack()
    label3.config(text='Recall: '+str(r))
    label3.pack()
    label4.config(text='Precision:'+str(p))
    label4.pack()
    label5 = Label(root,bg='white', text='=============================================')
    label5.pack()
    l.append(a)
    l.append(r)
    l.append(p)
    return l
    
    


# In[94]:


import urllib.request
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
root = Tk() 

# Adjust size 
root.geometry("1800x1400")
root.configure(bg='white')
root.title("Detecting security attacks in IoT network using deep learning model") 

global instruction1,instruction2,instructio1,instructin1
instruction1 = Label(root, bg='white',text = "Detecting security attacks in IoT network using deep learning model", font = ('Helvetica', 12, "bold"), fg="red") 
instruction1.pack() 
instructio1 = Label(root, bg='white',text = "", font = ('Helvetica', 12, "bold"), fg="red") 
instructio1.pack() 
instructin1 = Label(root, bg='white',text = "", font = ('Helvetica', 12, "bold"), fg="red") 
instructin1.pack() 
instruction2 = Label(root, bg='white',text = "Browse and select a CSV file as input to automatically start the process", font = ('Helvetica', 10, "bold"), fg="black") 
instruction2.pack()  
exit1=Button(root, bg='white',text="Exit", padx=20, command=root.destroy).place(anchor=W,x=725, y=500) 
#adv=Button(root,bg='white', text="Start", padx=20, command=main).place(anchor=W,x=400, y=300)
button_explore = Button(root,text = "Browse Files",bg='white', padx=20,command = browseFiles).place(anchor=W,x=700, y=300)
#Button(root, text="Know More", command=info).place(anchor=S, x=200, y=250)

root.mainloop() 


# In[ ]:




