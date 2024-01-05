#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[3]:


df=pd.read_csv("Dataset_Path")


# In[4]:


print(df)


# In[5]:


data = df.where((pd.notnull(df)), '')


# In[6]:


data.head(10)


# In[7]:


data.info()


# In[8]:


data.shape


# In[9]:


data.loc[data['Category'] == 'spam', 'Category',]=0
data.loc[data['Category'] == 'ham', 'Category',]=1


# In[10]:


X = data['Message']
Y = data['Category']


# In[11]:


print(X)


# In[12]:


print(Y)


# In[13]:


X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2, random_state = 3)


# In[14]:


print(X.shape)
print(X_train.shape)
print(X_test.shape) 


# In[15]:


print(Y.shape)
print(Y_train.shape)
print(Y_test.shape) 


# In[18]:


from sklearn.feature_extraction.text import TfidfVectorizer

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[19]:


print(X_train)


# In[20]:


print(X_train_features )


# In[21]:


model = LogisticRegression()


# In[23]:


model.fit(X_train_features, Y_train)


# In[24]:


prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train,prediction_on_training_data)


# In[25]:


print('Accuracy on training data : ', accuracy_on_training_data)


# In[26]:


prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test,prediction_on_test_data)


# In[33]:


print('Accuracy on test data:', accuracy_on_test_data)


# In[34]:


input_your_mail = ["this is the 2nd time we have tried to contact u. U have won the 400 prize.2 clam is easy,just cal 09923474"]

input_data_features = feature_extraction.transform(input_your_mail)

prediction = model.predict(input_data_features)
print(prediction)
if(prediction[0]==1):
    print('Ham mail')
else:
    print('spam mail')


# In[35]:


input_your_mail = ["hi how are you"]
input_data_features = feature_extraction.transform(input_your_mail)

prediction = model.predict(input_data_features)
print(prediction)

if prediction[0] == 1:
    print('Ham mail')
else:
    print('Spam mail')

