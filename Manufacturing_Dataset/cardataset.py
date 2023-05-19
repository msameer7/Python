#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_csv("D://Universities Lectures//Ned University of ENG. AND TECH//Machine Learning//Updated(carevaluation)//car_evaluation.csv")


# In[2]:


# Giving Columns Name
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df.columns = col_names
col_names


# In[3]:


df.head()


# In[4]:


#Checking the Dtype and Null values
df.info()


# In[5]:


# Checking Shape
df.shape


# In[6]:


# check percentage of missing
df.isnull().mean()


# In[7]:


from sklearn.preprocessing import LabelEncoder
# Define the columns to encode
cols_to_encode = ["buying", "maint", "lug_boot", "safety","class","doors","persons"]

# Encode categorical columns
le = LabelEncoder()
for col in cols_to_encode:
    df[col] = le.fit_transform(df[col])


# In[8]:


df.head(566)


# In[9]:


df.tail(150)


# In[11]:


import streamlit as st

scatterplot = st.line_chart(df)


# In[12]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[13]:


X = df.drop("class", axis=1)
y = df["class"]


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


model = LogisticRegression()
model.fit(X_train, y_train)


# ### Below mentioned features are encode by LabelEncoder
# ** Persons 0 means 2 persons
# 
# ** Persons 2 means 1 persons
# 
# ** Persons 4 means 2 persons
# 
# ** Doors 0 means 2 Doors
# 
# ** Doors 2 means 3 Doors
# 
# ** Doors 3 means 4 Doors

# In[16]:


# Define the feature names and their encoded values
feature_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
encoded_values = {"buying": [3, 1, 0], "maint": [3, 0, 1], "doors": [0, 2, 3], "persons": [0, 1, 2], "lug_boot": [2, 1, 0], "safety": [2, 0, 1]}


# ## Taking Input for Prediction in Streamlit

# In[17]:


# Define the user input section
st.header("Car Evaluation Prediction")
st.write("Enter the car features below:")
user_input = {}
for feature in feature_names:
    user_input[feature] = st.selectbox(feature, encoded_values[feature])


# In[18]:


# Make the prediction
prediction = model.predict(pd.DataFrame(user_input, index=[0]))
st.write("Predicted class:", prediction[0])


# ## Streamlit Visualizations

# In[19]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[20]:


# Sidebar for selecting the chart type
chart_type = st.sidebar.selectbox("Select chart type", ["Count Plot", "Pie Chart", "Box Plot", "Violin Plot", "Heatmap"])

# Count plot
if chart_type == "Count Plot":
    column = st.selectbox("Select column", df.columns)
    sns.countplot(x=column, data=df)
    st.pyplot()

# Pie chart
elif chart_type == "Pie Chart":
    column = st.selectbox("Select column", df.columns)
    fig1, ax1 = plt.subplots()
    ax1.pie(df[column].value_counts(), labels=df[column].unique(), autopct='%1.1f%%', startangle=90)
    ax1.axis('equal') 
    st.pyplot(fig1)

# Box plot
elif chart_type == "Box Plot":
    column_x = st.selectbox("Select X column", df.columns)
    column_y = st.selectbox("Select Y column", df.columns)
    sns.boxplot(x=column_x, y=column_y, data=df)
    st.pyplot()

# Violin plot
elif chart_type == "Violin Plot":
    column_x = st.selectbox("Select X column", df.columns)
    column_y = st.selectbox("Select Y column", df.columns)
    sns.violinplot(x=column_x, y=column_y, data=df)
    st.pyplot()

# Heatmap
else:
    sns.heatmap(df.corr(), annot=True)
    st.pyplot()

