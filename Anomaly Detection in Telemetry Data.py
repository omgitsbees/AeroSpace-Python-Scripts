import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.svm import OneClassSVM 
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
import seaborn as sns

# Configuration
DATA_FILE = "telemetry_data.csv"
ANOMALY_REPORT = "anomaly_report.csv"
PLOTS_DIRECTORY = "plots/"
ANOMALY_THRESHOLD = 0.1 # Adjust based on desired sensitivty

# Step 1: Load and Preprocess Data
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['timestamp'])
    data.set_index('timestamp', inplace=True)
    return data

def preprocess_data(data):
    # Fill missing values
    data.fillna(method='ffill', inplace=True)
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# Step 2: Anomaly Detection Using Isolation Forest
def detect_anomalies_isolation_forest(data):
    iso_forest = IsolationForest(contamination=ANOMALY_THRESHOLD, random_state=42)
    data['anomaly_score'] = iso_forest.fit_predict(data)
    return data

# Step 3: Anomaly Detection Using One-Class SVM
def detect_anomalies_one_class_svm(data):
    oc_svm = OneCLassSVM(kernel='rbf', gamma=0.01, nu=ANOMALY_THRESHOLD)
    data['anomaly_score'] = oc_svm.fit_predict(data)
    return data

# Step 4: Dimensionality Reduction Using PCA
def apply_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    return principal_components

# Step 5: Visualization
def visualize_anomalies(data, principal_components):
    plt.figure(figsize=(10, 6))
    plt.scatter(principal_components[:, 0], principla_components[:, 1], c=data['anomaly_score'], cmap='coolwarm')
    plt.title('Anomaly Detection Visualization (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Anomaly Score')
    plt.savefig(PLOTS_DIRECTORY + 'anomaly_detection_pca.png')
    plt.show()

def plot_timeseries(data):
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=data, dashes=False)
    plt.title('Telemetry Data Time Series with Anomalies')
    plt.xlabel('Timestamp')
    plt.ylabel('Telemetry Values')
    anomaly_times = data[data['anomaly_score'] == -1].index
    for xc in anomaly_times:
        plt.axvline(x=xc, color='r', linestyle='--')
    plt.savefig(PLOTS_DIRECTORY = 'anomaly_timeseries.png')
    plt.show()

# Step 6: Report Generation
def generate_report(data):
    anomaly_data = data[data['anomaly_score'] == -1]
    anomaly_data.to_csv(ANOMALY_REPORT)
    print(f"Anomaly report generated: {ANOMALY_REPORT}")
    print(f"Total anomalies detected: {len(anomaly_data)}")

def generate_classification_report(data):
    y_true = np.zeros(len(data))
    y_pred = (data['anomaly_score'] == -1).astype(int)
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
    print(confusion_matrix(y_true, y_pred))

# Step 7: Automation and Integration
def main():
    # Load and preprocess data
    data = load_data(DATA_FILE)
    scaled_data = preprocess_data(data)

    # Anomaly detection using Isolation Forest
    data_with_anomalies = detect_anomalies_isolation_forest(pd.DataFrame(scaled_data, index=data.index, columns=data.columns))

    # Apply PCA for visualization
    principal_components = apply_pca(scaled_data)

    # Visualize anomalies
    visualize_anomalies(data_with_anomalies, principal_components)

    # Generate anomaly report
    generate_report(data_with_anomalies)
    generate_classification_report(data_with_anomalies)

# Execution
if __name__ == "__main__":
    main()