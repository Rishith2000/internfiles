#!/usr/bin/env python
# coding: utf-8

# In[118]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
df=pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
print(df.shape)
print(df.describe)
x=df['Hours']
y=df['Scores']
x=x.values.reshape(-1,1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
plt.title('Hours vs Scores')
plt.scatter(x_test,y_test)
r=linear_model.LinearRegression()
r.fit(x_train,y_train)
yp=r.predict(x_test)
plt.plot(x_test,yp,color='red')
plt.show
r.predict([[9.25]])


# In[ ]:





# In[115]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
df=pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
print(df.shape)
print(df.describe)
x=df['Hours']
y=df['Scores']
x=x.values.reshape(-1,1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
plt.title('Hours vs Scores')
plt.scatter(x_test,y_test)
r=linear_model.LinearRegression()
r.fit(x_train,y_train)
yp=r.predict(x_test)
plt.plot(x_test,yp,color='red')
plt.show
r.predict([[9.25]])




# In[105]:





# In[109]:





# In[ ]:





# In[ ]:





# In[ ]:




