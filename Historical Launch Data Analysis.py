import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from datetime import datetime

# Generating a synthetic dataset for historical launch data
np.random.seed(42)
n_samples = 1000

data = {
    'Launch_Date': pd.date_range(start='2000-01-01', periods=n_samples, freq='M'),
    'Launch_Mass': np.random.randint(500, 6000, size=n_samples),
    'Payload_Mass': np.random.randint(100, 3000, size=n_samples),
    'Launch_Success': np.random.choice([1, 0], size=n_samples, p=[0.8, 0.2])
}

launch_data = pd.DataFrame(data)
launch_data['Year'] = launch_data['Launch_Date'].dt.year
launch_data['Month'] = launch_data['Launch_Date'].dt.month
launch_data['Day'] = launch_data['Launch_Date'].dt.day
launch_data['Payload_to_LaunchMass_Ratio'] = launch_data['Payload_Mass'] / launch_data['Launch_Mass']

# Exploratory Data Analysis
plt.figure(figsize=(14, 7))
sns.countplot(data=launch_data, x='Year', hue='Launch_Success')
plt.title('Yearly Launch Success vs Failure')
plt.show()

plt.figure(figsize=(14, 7))
sns.boxplot(data=launch_data, x='Launch_Success', y='Payload_to_LaunchMass_Ratio')
plt.title('Payload to Launch Mass Ratio vs Launch Success')
plt.show()

# Data Correlation Matrix
corr_matrix = launch_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Preparing Data for Machine Learning
features = ['Year', 'Month', 'Day', 'Launch_Mass', 'Payload_Mass', 'Payload_to_LaunchMass_Ratio']
X = launch_data[features]
y = launch_data['Launch_Success']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA for Dimensionality Reduction
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# RandomForest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_pca, y_train)

# Model Evaluation
y_pred = model.predict(X_test_pca)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# KMeans Clustering for Insight
kmeans = KMeans(n_clusters=2, random_state=42)
launch_data['Cluster'] = kmeans.fit_predict(X_train_pca)

# Visualizing Clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=launch_data['Cluster'])
plt.title('KMeans Clustering of Launch Data')
plt.show()

# Predicting Launch Success for New Data
new_data = pd.DataFrame({
    'Year': [2024],
    'Month': [9],
    'Day': [3],
    'Launch_Mass': [550],
    'Payload_Mass': [150],
    'Payload_to_LaunchMass_Ratio': [150/550]
})
new_data_scaled = scaler.transform(new_data)
new_data_pca = pca.transform(new_data_scaled)
success_prob = model.predict_proba(new_data_pca)[:, 1]

print(f"Predicted Launch Success Probability: {success_prob[0]:.2f}")

# Time Series Analysis
launch_data.set_index('Launch_Date', inplace=True)
launch_data['Launch_Success'].resample('M').mean().plot(figsize=(14, 7))
plt.title('Monthly Launch Success Rate')
plt.show()

# Analyzing Trends Over Time
monthly_trends = launch_data['Launch_Success'].resample('M').mean()
monthly_trends.rolling(window=12).mean().plot(figsize=(14, 7), color='red')
plt.title('Rolling Average of Launch Success Rate')
plt.show()

# Anomaly Detection in Launch Data
rolling_mean = monthly_trends.rolling(window=12).mean()
rolling_std = monthly_trends.rolling(window=12).std()
anomalies = monthly_trends[(monthly_trends > rolling_mean + 2 * rolling_std) |
                           (monthly_trends < rolling_mean - 2 * rolling_std)]

plt.figure(figsize=(14, 7))
plt.plot(monthly_trends, label='Monthly Launch Success Rate')
plt.plot(rolling_mean, color='red', label='Rolling Mean')
plt.scatter(anomalies.index, anomalies, color='red', label='Anomalies', marker='o')
plt.title('Anomaly Detection in Launch Success Rate')
plt.legend()
plt.show()
