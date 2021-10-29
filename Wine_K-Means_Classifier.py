# Rafael Espinosa Mena
# K-means clustering for classification
# 2021

import pandas as pd
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Read dataset into dataframe
df = pd.read_csv (r'/Users/rafaelespinosa/Documents/ITP499/HWs/hw1/wineQualityReds.csv')
pd.set_option("display.max_columns", None)

# Drop "wine" from dataframe given it is not important for predictions
df.drop(df.columns[0], axis = 1, inplace = True)

# Extract Quality and store it as the target matrix
qualityDf = df[['quality']]

# Drop quality to create the predictor matrix
df.drop('quality', axis=1, inplace=True)

# Print dataframe and quality
print("5) Full Data:\n", df)
print("5) Quality:\n", qualityDf, "\n")

# Normalize dataframe
norm = Normalizer()
df_norm = pd.DataFrame(norm.transform(df), columns=df.columns)

# Print normalized dataframe
print("7) Normalized dataframe:\n", df_norm)

# Create a range of k values from 1:11
ks = range(1, 11)
inertias = []
for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(df_norm)
    inertias.append(model.inertia_)

# Plot the chart for various k's
plt.plot(ks, inertias, "-o")
plt.xlabel("Number of Clusters, k")
plt.ylabel("Inertia")
plt.xticks(ks)
plt.show()

# 11) k clusters, using k=6
model = KMeans(n_clusters = 6, random_state = 2021)
model.fit(df_norm)
labels = model.predict(df_norm)
df_norm["Cluster label"] = pd.Series(labels)

# Add quality back to the dataframe
df_norm['quality'] = qualityDf

# Print Crosstab
cTab = pd.crosstab(index = df_norm['quality'], columns = df_norm['Cluster label'])
print("13. Crosstab:\n", cTab)



