#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[5]:


dir1=r"D:\LinearRegrassion.csv"


# In[7]:


df=pd.read_csv(dir1)
df.head()


# In[64]:


x=df[['Feature 3']].values
y=df[['Target']].values


# In[65]:


print(x)
print(y)


# In[66]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=.9,random_state=1)
print(x_train,y_train,x_test,y_test)


# In[67]:


from sklearn.linear_model import LinearRegression
li=LinearRegression()
li.fit(x_train,y_train)


# In[ ]:





# In[68]:


y_test_pre=li.predict(x_test)
y_train_pre=li.predict(x_train)
print(y_test_pre)
print(y_train_pre)


# In[69]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print(r2_score(y_train,y_train_pre))
print(r2_score(y_test,y_test_pre))


# In[70]:


m=li.coef_
c=li.intercept_
print(m,c)


# In[47]:


n=m*2+c
print(n)


# In[43]:


plt.scatter(x_train,y_train,color='green',marker='*')
plt.plot(x_train,m*x_train+c,color='red')


# In[ ]:





# In[ ]:




