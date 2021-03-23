#!/usr/bin/env python
# coding: utf-8

# In[50]:


#GEOP_592 Assigment_3
#Codes link: https://www.datarmatics.com/data-science/how-to-perform-logistic-regression-in-pythonstep-by-step/
#We are Provided with a data for 6800 micro-earthquake waveforms. 100 features have been extracted form it. 
#We need to perform logistic regression analysis to find that if it is a micro-earthquake (1) or noise (0).


# In[1]:


#Import the module
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np


# In[2]:


label = np.load('label.npy')


# In[3]:


features = np.load('features.npy')


# In[4]:


features.shape


# In[5]:


label.shape


# In[10]:


label[1:10]


# In[20]:


features[0:10,0:10]


# In[22]:


# Generate and dataset for Logistic Regression, This is given in the tutoriL, Not for our use in this assignmnet
x, y = make_classification(
    n_samples=100,
    n_features=2,
    n_classes=2,
    n_clusters_per_class=1,
    flip_y=0.03,
    n_informative=1,
    n_redundant=0,
    n_repeated=0
)
print(y)


# In[27]:


y.shape


# In[28]:


x.shape


# In[29]:


# Create a scatter plot
plt.scatter(x[:,1], y, c=y, cmap='rainbow')
plt.title('Scatter Plot of Logistic Regression')
plt.show()


# In[30]:


# Split the dataset into training and test dataset: Taking 5500 out of 6800 data points for training set, so 1300 will be used for test data points. 
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size = 1300, random_state=1)


# In[35]:


# Create a Logistic Regression Object, perform Logistic Regression
log_reg = LogisticRegression(solver='lbfgs',max_iter=500)
log_reg.fit(x_train, y_train)


# In[36]:


# Show to Coeficient and Intercept
print(log_reg.coef_)
print(log_reg.intercept_)


# In[37]:


# Perform prediction using the test dataset
y_pred = log_reg.predict(x_test)


# In[38]:


# Show the Confusion Matrix
confusion_matrix(y_test, y_pred)


# In[40]:


x_test.shape


# In[41]:


y_test.shape


# In[42]:


x_train.shape


# In[43]:


y_train.shape


# In[45]:


print(log_reg.score(x_test, y_test))


# In[49]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:


# So, our classification for '0' and '1' is correct for 99% & 98% respectively with overall accuracy of 98%. 

