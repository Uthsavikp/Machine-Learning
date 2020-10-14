#!/usr/bin/env python
# coding: utf-8

# # Name : UTHSAVI KP

# # Task-1 : Prediction using Supervised ML
In this task we are supposed to predict the percentage of marks student's will score based on the number of hours they study.

This is a simple linear regression task as it involves just 2 variables.

What will be predicted score if a student studies for 9.25 hrs/ day?
# In[4]:


# Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


# Reading the data from the url
df_data = pd.read_csv("http://bit.ly/w-data")


# In[7]:


df_data.head(5)


# In[8]:


df_data.tail()


# In[9]:


df_data.shape


# In[10]:


df_data.describe()


# In[11]:


df_data.info()


# In[12]:


df_data.isnull().sum()


# In[13]:


plt.figure(figsize=(10,6))
plt.scatter(df_data['Hours'],df_data['Scores'],label='Data')
plt.title('No. of hours studied vs marks scored')
plt.xlabel('Hours studied')
plt.ylabel('Percentage scored')
plt.legend()
plt.show()


# #### By looking at the above graph, we can obsorve that there is a positive linear relationship between the number of hours 
# #### a student studied and marks scored by the student.
#  

# # Heatmap for correlation check

# In[14]:


import seaborn as sns
k = 2
corrmat = df_data.corr()
columns = corrmat.nlargest(k,'Hours')['Hours'].index
cm = np.corrcoef(df_data[columns].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm,cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=columns.values,
xticklabels=columns.values)
plt.show()


# #### From the above heatmap,we can see that there is a positive correlation between Hours and Score
# #### i.e as the number of hours studied increases marks scored by the student also increases

# ## Data Preparation

# In[15]:


# Divide the data into features(X) and labels(y)
X = df_data.drop(['Scores'],axis=1)
y = df_data['Scores'] 


# In[16]:


# From sklearn model selection importing the function train_test_split for diving the data into training and testing dataset 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)


# In[20]:


y_train.shape


# In[21]:


y_test.shape


# # Training with Linear Regression Algorithm

# In[22]:


from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train,y_train)
#lr_predict = lr.predict(X_test)


# ### To retrive the intercept

# In[23]:


print(regression.intercept_)


# ### For retriving the slope (coefficient of X)

# In[24]:


print(regression.coef_)


# In[25]:


#Plotting the regression line on full dataset
line = regression.coef_*X + regression.intercept_

plt.figure(figsize=(10,6))
plt.scatter(X, y,s=100)
plt.title('No. of hours studied vs marks scored')
plt.xlabel('Hours studied')
plt.ylabel('Marks scored')
plt.plot(X, line,color='red', label='Regression Line')
plt.legend()
plt.show()


# # Making the Prediction

# In[19]:


# Making prediction using the test dataset
y_pred = regression.predict(X_test)


# In[20]:


#To compare the actual output of X_test with the predicted X_test


# In[21]:


df_data = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
df_data


# ## Plotting the Regression line on Test data

# In[22]:


plt.figure(figsize=(10,6))
plt.scatter(X_test,y_test,s=100,color='g',label='Actual Score')
plt.scatter(X_test,y_pred,s=100,color='c',label='Predicted Score')
plt.plot(X_test,y_pred,color='r',label='Regression line')
plt.title('No. of hours studied vs marks scored')
plt.xlabel('Hours studied')
plt.ylabel('Percentage scored')
plt.legend()
plt.show()


# # Evaluating the Model Performance

# In[23]:


from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:',metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print('R2 score:',metrics.r2_score(y_test,y_pred))


# #### R-squared represents the proportion of the variance for a dependent variable that is explained by an independent variable.
# #### Here the R-squared is 0.9367,it means that a full 93.67% of the variation of one variable is completely explained by the other. 
# 

# ## What will be predicted score if a student studies for 9.25 hrs/ day?

# In[24]:


hours = 9.25
test = np.array([hours]).reshape(-1, 1)
score_pred = regression.predict(test)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(np.round(score_pred[0],2)))

