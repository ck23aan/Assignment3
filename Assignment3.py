#!/usr/bin/env python
# coding: utf-8

# ## GDP per capita (current USD)

# In[22]:


# https://data.worldbank.org/indicator/NY.GDP.PCAP.CD


# In[45]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd


# In[17]:


def worldbank_data(filename):
    """
    Ingests World Bank data from a CSV file.

    Parameters:
    - filename: The path to the CSV file.

    Returns:
    - df_byyears (DataFrame): Dataframe with rows representing years and columns representing countries.
    - df_bycountry (DataFrame): Original dataframe representing countries and columns representing years.
    """    
    df_bycountry = pd.read_csv(filename, skiprows=4)
    df_byyears = df_bycountry.transpose()
    df_byyears.columns = df_byyears.iloc[0]
    
    return df_byyears, df_bycountry

df_byyears, df_bycountry = worldbank_data('API_NY.GDP.PCAP.CD_DS2_en_csv_v2_6298251.csv')

df = df_bycountry[['Country Name', 'Indicator Name'] + list(map(str, range(1960, 2023)))]

df = df.dropna()

dfx = df[["Country Name", "1960"]].copy()

dfx.head()


# In[21]:


dfx["Growth"] = 100.0 * (df["2022"] - df["1960"]) / (df["1960"])
dfx = dfx.dropna()
dfx.describe()


# In[28]:


import matplotlib.pyplot as plt, seaborn as sns

plt.figure(figsize=(7, 5))
scatter_plot = plt.scatter(dfx["1960"], dfx["Growth"], 10, label="GDP per capita (current USD)")
plt.xlabel("GDP per capita (current USD) in 1960")
plt.ylabel("Growth Percentage in GDP per capita from 1960 to 2022")
plt.title("GDP per capita (current USD) vs. Growth Percentage 1960 -2022")
plt.legend()
plt.show()


# In[52]:


from sklearn.preprocessing import StandardScaler
import sklearn.metrics as skmet

def normalize_data(data_frame, features):
    """
    Function to normalize the specified features using StandardScaler.

    Parameters:
    - data_frame: The input DataFrame.
    - features: List of column names to be normalized.

    Returns:
    - Normalized DataFrame.
    - StandardScaler object for inverse transformation.
    """

    scaler = StandardScaler()
    subset_features = data_frame[features]
    scaler.fit(subset_features)
    normalized_data = scaler.transform(subset_features)

    # Create a DataFrame with normalized data
    normalized_df = pd.DataFrame(normalized_data, columns=features)

    return normalized_df, scaler


# In[53]:


def inverse_transform_data(normalized_df, scaler, features):
    """
    Function to inverse transform the normalized data back to the original scale.

    Parameters:
    - normalized_df: The normalized DataFrame.
    - scaler: StandardScaler object used for normalization.
    - features: List of column names to be inverse transformed.

    Returns:
    - DataFrame with inverse transformed data.
    """

    inverse_transformed_data = scaler.inverse_transform(normalized_df[features])

    return pd.DataFrame(inverse_transformed_data, columns=features)


# In[54]:


norm, scaler = normalize_data(dfx, ['1960', 'Growth'])


# ## K-Means++

# In[48]:


from sklearn.cluster import KMeans

def silhouette_score(xy, n):
    """
    Calculates silhouette score for n clusters using KMeans++ initialization.

    Parameters:
    - xy: Input data.
    - n: Number of clusters.

    Returns:
    Silhouette score.
    """
    kmeans = KMeans(n_clusters=n, init='k-means++', n_init=20)
    kmeans.fit(xy)
    labels = kmeans.labels_
    score = skmet.silhouette_score(xy, labels)
    return score


# In[49]:


for i in range(2, 10):
    score = silhouette_score(norm, i)
    print(f"The silhouette score for {i: 3d} is {score: 7.4f}")


# In[55]:


inv_df = inverse_transform_data(norm, scaler, ['1960', 'Growth'])


# In[58]:


kmeans = KMeans(n_clusters=4, init='k-means++', n_init=20)
kmeans.fit(norm)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
centroids_inv = inverse_transform_data(pd.DataFrame(centroids, columns=["1960", "Growth"]), scaler, ["1960", "Growth"])
xkmeans, ykmeans = centroids_inv["1960"], centroids_inv["Growth"]

plt.figure(figsize=(7, 5))
plt.scatter(dfx["1960"], dfx["Growth"], c=labels, marker="o", s=10)
plt.scatter(xkmeans, ykmeans, marker="x", c="red", s=50, label="Cluster Centroids")
plt.xlabel("GDP per capita (current USD) in 1960")
plt.ylabel("Growth Percentage in GDP per capita from 1960 to 2022")
plt.title("K-Means++ Cluster Representation of Data")
plt.legend()
plt.show()


# ## Curve Fit

# In[61]:


world_GDP = df_byyears.loc['1960':'2022', ['World']].reset_index().rename(columns={'index': 'Year', 'World': 'GDP per capita'})
world_GDP = world_GDP.apply(pd.to_numeric, errors='coerce')
world_GDP.describe()


# In[63]:


plt.figure(figsize=(8, 5))
sns.lineplot(data=world_GDP, x='Year', y='GDP per capita')
plt.xlabel('Year')
plt.ylabel('GDP per capita')
plt.title('GDP per capita across world between 1960-2022')
plt.show()


# In[64]:


def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    # makes it easier to get a guess for initial parameters
    t = t - 2010
    f = n0 * np.exp(g*t)
    return f


# In[68]:


import scipy.optimize as opt
import numpy as np


param, covar = opt.curve_fit(exponential, world_GDP["Year"], world_GDP["GDP per capita"], p0=(1.3e4, 0.1))
world_GDP["fit"] = exponential(world_GDP["Year"], *param)
plt.figure(figsize=(7, 5))
sns.lineplot(data=world_GDP, x="Year", y="GDP per capita", label="World GDP per capita")
sns.lineplot(data=world_GDP, x="Year", y="fit", label="Exponential Fit")
plt.xlabel("Year")
plt.ylabel('GDP per capita')
plt.title('GDP per capita across world between 1960-2022')
plt.legend()
plt.show()


# In[70]:


import errors
years = np.arange(2021, 2051, 1)
predictions = exponential(years, *param)
confidence_range = errors.error_prop(years, exponential, param, covar)


# In[75]:


plt.figure(figsize=(10, 6))
sns.lineplot(x= world_GDP["Year"], y= world_GDP["GDP per capita"], label="World GDP per capita")
sns.lineplot(x=years, y=predictions, label="Prediction", color='red')
plt.fill_between(years, predictions - confidence_range, predictions + confidence_range, 
                 color='yellow', alpha=0.4, label="Confidence Range")
plt.xlabel("Year")
plt.ylabel("GDP per capita")
plt.title("Exponential Growth of GDP per capita Prediction")
plt.legend()
plt.show()


# In[ ]:




