#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter


# In[2]:


Patient = pd.read_csv(r"C:\Users\srira\Desktop\Ram\Data science\Course - Assignments\Module 25 - Survival Analytics\Datasets\Patient.csv")
Patient.head()


# In[3]:


#EDA


# In[4]:


Patient.shape


# In[7]:


Patient['Scenario'].value_counts()


# In[8]:


#There are no multiple scenario's are group's and just an event type. We can goahead and build model


# In[11]:


T = Patient['Followup']


# In[10]:


kmf = KaplanMeierFitter()


# In[12]:


kmf.fit(T,event_observed=Patient['Eventtype'])


# In[13]:


kmf.percentile


# In[14]:


kmf.event_table


# In[15]:


kmf.plot()

