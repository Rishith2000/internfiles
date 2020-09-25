#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("creditcard[1].csv")


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df['Class'].value_counts()


# In[7]:


df.hist(figsize=(20,20))
plt.show()


# In[46]:


sns.countplot(df.Class)
plt.title("Class Distribution")


# In[48]:


corre=df.corr()
fig=plt.figure(figsize=(5,5))
sns.heatmap(corre)
plt.show()


# In[28]:


#Extracting Dependent and Independent variables
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
print(x.shape)
print(y.shape)


# In[29]:


x.head()


# In[30]:


y.head()


# In[32]:



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print("Train and test sizes, respectively:", len(X_train), len(y_train), "|", len(X_test), len(y_test))
print("Total number of frauds:", len(y.loc[df['Class'] == 1]))
print("Number of frauds on y_test:", len(y_test.loc[df['Class'] == 1]))
print("Number of frauds on y_train:", len(y_train.loc[df['Class'] == 1]))
X_org = x
y_org = y


# In[33]:


from sklearn.linear_model import LogisticRegression
lr =LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)


# In[69]:


yp=np.array(lr.predict(X_test))
y=np.array(y_test)


# In[70]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[73]:


a=confusion_matrix(y,yp)
print(a)
sns.heatmap(pd.DataFrame(a),annot=True,fmt='g')
plt.ylabel("Actual label")
plt.xlabel("Predictable label")
plt.show()


# In[37]:


print(accuracy_score(y,yp))


# In[38]:


labels=['Non-Fraud','Fraud']
print(classification_report(y_test,yp))


# In[39]:


from sklearn.metrics import roc_curve, auc
y_score = lr.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_score)

plt.title('ROC curve')
plt.xlabel('FPR (Precision)')
plt.ylabel('TPR (Recall)')

plt.plot(fpr,tpr)
plt.plot((0,1), ls='dashed',color='black')
plt.show()
print('Area under curve (AUC): ', auc(fpr,tpr))


# In[61]:


import imblearn
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X= X_org
y= y_org
X_res, y_res = sm.fit_resample(X, y)

#  Train and Test datasets
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_res, y_res, test_size=0.2, random_state=0)


# In[62]:


print("After oversampling,the shape os x_train:{}".format(X_train_s.shape))
print("After oversampling,the shape of y_train:{}".format(y_train_s.shape))
print("After oversampling,the count label '1':{}".format(sum(y_train_s==1)/len(y_train_s)*100.0,2))
print("After oversampling,the count label '0':{}".format(sum(y_train_s==0)/len(y_train_s)*100.0,2))


# In[67]:


sns.countplot(x=y_train_s,data=df)
plt.show()


# In[76]:


model = LogisticRegression(max_iter=1000)
model = model.fit(X_train, y_train)


# In[ ]:





# In[78]:


print("accuracy on the testing set:",model.score(X_test, y_test))

# Predict class labels for the test set
predicted = model.predict(X_test)

# Confusion Matrix for train data
b=confusion_matrix(predicted, y_test)
print(b)
sns.heatmap(pd.DataFrame(b),annot=True,fmt='g')
plt.ylabel("Actual label")
plt.xlabel("Predictable label")
plt.show()

print(classification_report(predicted, y_test))

#ROC, AUC
from sklearn.metrics import roc_curve, auc
y_score = model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_score)

plt.title('ROC curve')
plt.xlabel('FPR (Precision)')
plt.ylabel('TPR (Recall)')

plt.plot(fpr,tpr)
plt.plot((0,1), ls='dashed',color='black')
plt.show()
print('Area under curve (AUC): ', auc(fpr,tpr))


# In[ ]:




