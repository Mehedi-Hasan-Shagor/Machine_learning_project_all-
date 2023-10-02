#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB


# In[35]:


df=pd.read_csv('C:/Users/Naive.csv')
df


# In[34]:


Numerics=LabelEncoder()
inputs=df.drop('play',axis='columns')
target=df['play']
target
print(inputs)
print(target)


# In[47]:


inputs['weather_n']=Numerics.fit_transform(inputs['weather'])
print(inputs)
input_n=inputs.drop(['weather'],axis='columns')
input_n
print(inputs)


# In[46]:


Classifier=GaussianNB()
Classifier.fit(input_n,target)
Classifier.score(input_n,target)
probs = Classifier.predict_proba([[2]])
print(probs)


# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[9]:


dir1=r"D:\Naive-bias.csv"
df=pd.read_csv(dir1)
print(df)


# In[11]:


from sklearn.preprocessing import LabelEncoder
li=LabelEncoder()
for i in df.columns:
    df[i]=li.fit_transform(df[i])
print(df)


# In[13]:


test=df[['Outlook','Temp','Humidity','Windy']].values
target=df[['Play']]
print(test,target)


# In[16]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(test,target,test_size=0.2,random_state=1)
print(x_train,x_test,y_train,y_test)


# In[20]:


from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)
y_test_pre=nb.predict(x_test)
y_train_pre=nb.predict(x_train)
print(y_test_pre)


# In[21]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_test_pre))
print(accuracy_score(y_train,y_train_pre))


# In[30]:


y_pre=nb.predict([[1,1,1,0]])
print(y_pre)
nb.score([[1,1,1,0]],y_pre)
probs = nb.predict_proba([[1,1,1,0]])
print(probs)


# In[24]:





# In[ ]:




