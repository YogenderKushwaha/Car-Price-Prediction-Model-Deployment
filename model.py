#!/usr/bin/env python
# coding: utf-8

# # Automobile Dataset 

# In[3]:


# import libraries
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[4]:


df= pd.read_csv("AutoData.csv")
df.head()


# In[5]:


df.shape


# In[6]:


df.isnull().sum() #checking null values


# In[ ]:





# In[9]:





# # Train_test model

# In[13]:


from sklearn.model_selection import train_test_split

X=df[['enginesize','stroke','peakrpm','curbweight','carwidth','citympg']].values    
y = df['price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# # RandomForestRegressor

# In[14]:


from sklearn.ensemble import RandomForestRegressor 
  
 # create regressor object 
model = RandomForestRegressor(n_estimators = 100, random_state = 0) 
  
# fit the regressor with x and y data 
model.fit(X, y) 


# In[15]:


y_pred = model.predict(X_test)


# In[17]:


from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)*100
print('The R2 for this model is:', r2,'%')


# In[ ]:


import pickle 
pickle_out= open("model.pkl","wb")
pickle.dump(model, pickle_out)
pickle_out.close()


# In[ ]:




