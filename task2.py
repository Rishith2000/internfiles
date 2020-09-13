#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import seaborn as sns


# In[43]:


df=pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
df.head()


# In[44]:


df.shape


# In[45]:


df.describe()


# In[46]:


plt.scatter(df.Hours,df.Scores)
plt.title('Hours vs Scores')
plt.xlabel('Hours')
plt.ylabel('Scores')


# In[47]:


sns.pairplot(df)


# In[48]:


x=df['Hours']
y=df['Scores']
x=x.values.reshape(-1,1)
y=y.values.reshape(-1,1)


# In[49]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[50]:



r=linear_model.LinearRegression()
r.fit(x_train,y_train)


# In[51]:



yp=r.predict(x_test)
y2=r.coef_*x+r.intercept_
plt.scatter(x,y)
plt.plot(x,y2,color='red')
plt.show


# In[52]:


r.predict([[9.25]])


# In[ ]:




