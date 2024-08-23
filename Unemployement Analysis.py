#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from IPython.display import HTML
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# In[10]:


df1 = pd.read_csv('Unemployment in India.csv')
df2 = pd.read_csv('Unemployment_Rate_upto_11_2020.csv')


# # Unemployment In India (2019)

# In[11]:


df1.head()


# In[12]:


df1.columns


# In[13]:


# Understanding the Data


# In[14]:


df1.info()


# In[19]:


# Rename columns for clarity
df1.columns = [ 'Region','Date', 'Frequency','Estimated Unemployment Rate','Estimated Employed',
               
               'Labour Participation Rate','Area']


# In[18]:


df1.head()


# In[89]:


df1.describe()


# # EDA

# In[15]:


# Handling Missing Values


# In[90]:


df1.isnull().sum()


# In[91]:


df1.shape


# In[92]:


df1=df1.dropna()


# In[93]:


df1.isnull().sum()


# In[94]:


# Convert the Date column to datetime
df1['Date'] = pd.to_datetime(df1['Date'])

# Extract the month number
df1['Month'] = df1['Date'].dt.month

# Extract the month name
df1['Month_Name'] = df1['Date'].dt.strftime('%B')


# In[95]:


# Statistics by Region
region_stats = df1.groupby (['Region'])[['Estimated Unemployment Rate', 'Estimated Employed', 'Labour Participation Rate']].mean().reset_index()
region_stats = round(region_stats, 2)
region_stats


# In[139]:


df1.pivot_table(values = 'Estimated Unemployment Rate', index ='Region',aggfunc=np.sum)


# In[29]:


# Calculate the average unemployment rate by state
average_unemployment_rate = df1.groupby('Region')['Estimated Unemployment Rate'].mean()

# Find the Region with the highest unemployment rate
Region_with_highest_unemployment = average_unemployment_rate.idxmax()
Region_highest_unemployment_rate = average_unemployment_rate.max()

# Find the Region with the lowest unemployment rate
Region_with_lowest_unemployment = average_unemployment_rate.idxmin()
lowest_unemployment_rate = average_unemployment_rate.min()

# Print the results
print("Region with the highest unemployment rate:", Region_with_highest_unemployment)
print("Highest unemployment rate:", Region_highest_unemployment_rate)
print("Region with the lowest unemployment rate:", Region_with_lowest_unemployment)
print("Lowest unemployment rate:", lowest_unemployment_rate)


# In[23]:


# Correlation Heatmap


# Ensure 'Month' is numeric if it's not already
heat_maps = df1[['Estimated Unemployment Rate', 'Estimated Employed', 'Labour Participation Rate', 'Month']]
heat_maps['Month'] = pd.to_numeric(heat_maps['Month'], errors='coerce')  # Convert 'Month' to numeric, if needed

# Drop rows with NaN values if any (optional)
heat_maps = heat_maps.dropna()

plt.figure(figsize=(10, 6))
sns.heatmap(heat_maps.corr(), annot=True, cmap='YlOrRd')  # Using correlation matrix for better visualization
plt.title('Heatmap of Statistics by Month')
plt.show()


# In[24]:


sns.set_style('darkgrid')
palette_color = ['blue', 'green']



# In[ ]:





# In[45]:


plt.figure(figsize=(15,10))
sns.countplot(y="Region",data=df1)
plt.show()


# In[113]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='Region', y='Estimated Unemployment Rate', data=df1,palette="brg")
plt.title("estimated_employee")
plt.xlabel("Region")
plt.ylabel("estimated_employee")
plt.xticks(rotation=90)
plt.show()


# In[115]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='Region', y='Labour Participation Rate', data=df1,palette="hsv")
plt.title("Labour_Participation_ Rate")
plt.xlabel("Region")
plt.ylabel("Labour_Participation_ Rate")
plt.xticks(rotation=90)
plt.show()


# In[116]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='Region', y='Estimated Employed', data=df1,palette="brg")
plt.title("estimated_employee")
plt.xlabel("Region")
plt.ylabel("estimated_employee")
plt.xticks(rotation=90)
plt.show()


# In[118]:


sns.histplot(data=df1, x="Estimated Unemployment Rate", kde=True,color="navy")
plt.show()


# In[119]:


sns.histplot(data=df1, x="Estimated Employed", kde=True,color="brown")
plt.show()


# In[121]:


sns.histplot(data=df1, x="Labour Participation Rate", kde=True,color="darkorange")
plt.show()


# # Unemployment Rate vs. Labor Participation Rate¶

# In[123]:


plt.figure(figsize=(10,7))
sns.scatterplot(data=df1, x="Estimated Unemployment Rate", y="Labour Participation Rate", hue="Region")
plt.show()


# In[124]:


plt.figure(figsize=(10,7))
sns.violinplot(x=df1["Estimated Unemployment Rate"])
plt.show()


# In[131]:


pair=df1[["Estimated Unemployment Rate","Estimated Employed","Labour Participation Rate"]]
sns.pairplot(pair ,markers="*", palette="green")
plt.show()


# In[132]:


sns.countplot(x="Area",data=df1)
plt.show()


# In[133]:


df1["Area"].value_counts()


# In[135]:


area_summary = df1.groupby('Area').agg({
    'Estimated Unemployment Rate': 'mean',
    'Estimated Employed': 'sum',
    'Labour Participation Rate': 'mean'
}).reset_index()
area_summary


# In[136]:


sns.barplot(x='Area', y='Estimated Employed', data=area_summary)


# # Unemployment in India (2020)
# 

# In[154]:


df2.head()


# In[156]:


df2.isnull().sum()


# In[155]:


df2.shape


# In[184]:


df2.columns = [ 'Region', 'Date', 'Frequency','Estimated Unemployment Rate','Estimated Employed',
               
               'Labour Participation Rate','Region.1','longitude','latitude']


# In[180]:


df2.info()


# In[181]:


df2.describe()


# In[182]:


plt.figure(figsize=(10,7))
sns.countplot(y="Region",data=df2)
plt.show()


# In[183]:


import plotly.express as px
fig = px.bar(df2, x="Region", y=" Estimated Employed", title=" Estimated_Employed", 
             animation_frame=' Date',template='plotly',color="Region.1")
fig.show()


# In[185]:


# Find the row with the maximum unemployment rate
max_unemployment_row = df2[df2['Estimated Unemployment Rate'] == df2['Estimated Unemployment Rate'].max()]

# Display the region, date, and unemployment rate
print(max_unemployment_row[['Region', 'Date', 'Estimated Unemployment Rate']])


# In[ ]:


#Which Region Has the Highest Unemployment Rate?



# In[186]:


# Group by region and calculate summary statistics
participation_stats = df2.groupby('Region')['Labour Participation Rate'].agg(['mean', 'min', 'max'])

# Display the statistics
print(participation_stats)


# In[187]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='Region', y='Labour Participation Rate', data=df2)
plt.xticks(rotation=90)  # Rotate x labels for better readability
plt.title('Distribution of Labour Participation Rates by Region')
plt.show()


# In[ ]:




