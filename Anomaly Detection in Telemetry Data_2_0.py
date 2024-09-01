import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Example: Load data from a CSV file
data_file_path = "path_to_your_telemetry_data.csv"
data = pd.read_csv(data_file_path)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging. getLogger('AerospaceTelemetryAnomalyDetection')

# Configuration
CONFIG = {
    'DATA_POINTS': 10000,
    'SEED': 42,
    'ANOMALY_FRACTION': 0.01,
    'FEATURES': ['altitude', 'velocity', 'acceleration', 'temperature', 'pressure', 'fuel_level'],
    'PLOT_DIR': 'plots',
    'REPORT_DIR': 'reports',
    'MODEL_DIR': 'models',
    'THRESHOLDS': {
        'z_score': 3,
        'iqr': 1.5,
        'isolation_forest': 0.05,
        'one_class_svm': 0.05,
        'autoencoder': 0.01
    }
}

# Ensure directories exist
os.makedirs(CONFIG['PLOT_DIR'], exist_ok=True)
os.makedirs(CONFIG['REPORT_DIR'], exist_ok=True)
os.makedirs(CONFIG['MODEL_DIR'], exist_ok=True)

def generate_synthetic_data(n_points, features, anomaly_fraction, seed):
    """
    Generates synthetic aerospace telemetry data with injected anomalies.
    """
    np.random.seed(seed)
    timestamps = pd.date_range(start=datetime.datetime.now(), periods=n_points, freq='S')
    data = pd.DataFrame(index=timestamps)

    # Generate normal data
    data['altitude'] = np.random.normal(loc=10000, scale=300, size=n_points) # in meters
    data['velocity'] = np.random.normal(loc=250, scale=10, size=n_points) # in m/s
    data['acceleration'] = np.random.normal(loc=0, scale=2, size=n_points) # in m/s^2
    data['temperature'] = np.random.normal(loc=15, scale=5, size=n_points) # in Celsius
    data['pressure'] = np.random.normal(loc=1013, scale=5, size=n_points) # in hPa
    data['fuel_level'] = np.random.normal(loc=5000, scale=200, size=n_points) # in liters

    # Inject anomalies
    n_anomalies = int(n_points * anomaly_fraction)
    anomaly_indices = np.random.choice(n_points, n_anomalies, replace=False)

    # Define anomalies for each feature
    data.loc[data.index[anomaly_indices], 'altitude'] *= np.random.choice([0.5, 1.5], size=n_anomalies)
    data.loc[data.index[anomaly_indices], 'velocity'] *= np.random.choice([0.5, 1.5], size=n_anomalies)
    data.loc[data.index[anomaly_indices], 'acceleration'] *= np.random.choice([-20, 20], size=n_anomalies)
    data.loc[data.index[anomaly_indices], 'temperature'] *= np.random.choice([-30, 30], size=n_anomalies)
    data.loc[data.index[anomaly_indices], 'pressure'] *= np.random.choice([-100, 100], size=n_anomalies)
    data.loc[data.index[anomaly_indices], 'fuel_level'] *= np.random.choice([0.1, 2], size=n_anomalies)

    # Create labels
    labels = np.zeros(n_points)
    labels[anomaly_indices] = 1 # 1 indicates anomaly

    data['label'] = labels

    logger.info(f"Synthetic data generated with {n_points} points and {n_anomalies} anomalies.")
    return data

def preprocess_data(data, features):
    """
    Preprocesses the data by handling missing values and scaling.
    """
    # Handle missing values
    data_clean = data.copy()
    data_clean = data_clean.fillna(method='ffill').fillna(method='bfill')

    # Scale data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_clean[features])

    logger.info("Data preprocessing completed.")
    return data_scaled, scaler

def detect_anomalies_z_score(data, threshold=3):
    """
    Detects anomalies using the Z-Score method.
    """
    z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
    anomalies = (z_scores > threshold).any(axis=1)
    logger.info(f"Z-Score detection completed with threshold {threshold}.")
    return anomalies

def detect_anomalies_iqr(data, threshold=1.5):
    """
    Detects anomalies using the Interquartile Range (IQR) method.
    """
    Q1 = no.percentile(data, 25, axis=0)
    Q3 = np.percentile(data, 75, axis=0)
    IQR = Q3 - Q1
    is_outlier = ((data < (Q1 - threshold * IQR)) | (data > (Q3 + threshold * IQR))).any(axis=1)
    logger.info(f"IQR detection completed with threshold {threshold}.")
    return is_outlier

def detect_anomalies_isolation_forest(data, contamination=0.01):
    """
    Detects anomalies using Isolation Forest.
    """
    iso_forest = IsolationForest(contamination=contamination, random_state=CONFIG['SEED'])
    preds = iso_forest.fit_predict(data)
    anomalies = preds == -1
    logger.info(f"Isolation Forest detection completed with contamination {contamination}.")
    return anomalies, iso_forest

def detect_anomalies_one_class_svm(data, nu=0.01):
    """
    Detects anomalies using One-Class SVM.
    """
    oc_svm = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
    preds = oc_svm.fit_predict(data)
    anomalies = preds == -1
    logger.info(f"One-Class SVM detection completed with nu {nu}.")
    return anomalies, oc_svm

def build_autoencoder(input_dim):
    """
    Builds and compiles an autoencoder model.
    """
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(16, activation='relu')(input_layer)
    encoder = Dense(8, activation='relu')(encoder)
    decoder = Dense(16, activation='relu')(encoder)
    decoder = Dense(input_dim, activation="linear")(decoder)
    autoencoder = Model(inpuits=input_layer, outputs=decoder)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    logger.info("Autoencoder model built and compiled.")
    return autoencoder

def detect_anomalies_autoencoder(data, threshold_ratio=1.5, epochs=50, batch_size=32):
    """
    Detects anomalies using Autoencoder.
    """
    x_train, x_test = train_test_split(data, test_size=0.2, random_state=CONFIG['SEED'])
    autoencoder = build_autoencoder(x_train.shape[1])
    history = autoencoder.fit(
        x_train, x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test),
        verbose=0
    )
    reconstructions = autoencoder.predict(data)
    reconstruction_errors = np.mean(np.square(data - reconstructions), axis=1)
    threshold = np.percentile(reconstruction_errors, 100 * (1 - CONFIG['THRESHOLDS']['autoencoder']))
    anomalies = reconstruction_errors > threshold
    logger.info(f"Autoencoder detection completed with threshold {threshold:.4f}.")
    return anomalies, autoencoder, treshold

def plot_anomalies(data, anomalies, method_name):
    """
    Plots the telemetry data highlighting the anomalies.
    """
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(CONFIG['FEATURES']):
        plt.subplot(len(CONFIG['FEATURES']), 1, i+1)
        plt.plot(data,index, data[feature], label=feature)
        plt.scatter(data.index[anomalies], data[feature][anomalies], color='r', label='Anomaly')
        plt.legend(loc='upper left')
        plt.tight_layout()
    plot_path = os.path.join(CONFIG['PLOT_DIR'], f'{method_name}_anomalies.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Anomaly plot saved to {plot_path}.")

def generate_report(data, anomalies, method_name):
    """
    Generates a report summarizing the anomalies detected.
    """
    anomaly_data = data[anomalies]
    report_path = os.path.join(CONFIG['REPORT_DIR'], f'{method_name}_anomaly_report.csv')
    anomaly_data.to_csv(report_path)
    logger.info(f"Anomaly report saved to {report_path}.")

def evaluate_detection(true_labels, predicted_anomalies, method_name):
    """
    Evaluates the anomaly detection performance.
    """
    report = classification_report(true_labels, predicted_anomalies, target_names=['Normal', 'Anomaly'])
    confusion = confusion_matrix(true_labels, predicted_anomalies)
    logger.info(f"Classification Report for {method_name}:\n{report}")
    logger.info(f"Confusion Matrix for {method_name}:\n{confusion}")

def main():
    # Generate synthetic data
    data = generate_synthetic_data(
        n_points=CONFIG['DATA_POINTS'],
        features=CONFIG['FEATURES'],
        anomaly_fraction=CONFIG['ANOMALY_FRACTION'],
        seed=CONFIG['SEED']
    )

# Preprocess data
data_scaled, scaler = preprocess_data(data, CONFIG['FEATURES'])

# Z-Score Method
z_score_anomalies = detect_anomalies_z_score(data_scaled, threshold=CONFIG['THRESHOLDS']['z_score'])
plot_anomalies(data, z_score_anomalies, 'z_score')
generate_report(data, z_score_anomalies, 'z_score')
evaluate_detection(data['label'], z_score_anomalies, 'Z-Score Method')

# IQR Method
iqr_anomalies = detect_anomalies_iqr(data_scaled, threshold=CONFIG['THRESHOLDS']['iqr'])
plot_anomalies(data, iqr_anomalies, 'iqr')
generate_report(data, iqr_anomalies, 'iqr')
evaluate_detection(data['label'], iqr_anomalies, 'IQR Method')

# Isolation Forest
iso_forest_anomalies, iso_forest_model = detect_anomalies_isolation_forest(
    data_scaled, contamination=CONFIG['THRESHOLDS']['isolation_forest']
)
plot.anomalies(data, iso_forest_anomalies, 'isolation_forest')
generate_report(data, iso_forest_anomalies, 'isolation_forest')
evaluate_detection(data['label'], iso_forest_anomalies, 'Isolation Forest')

# One-Class SVM
oc_svm_anomalies, oc_svm_model = detect_anomalies_one_class_svm(
    data_scaled, nu=CONFIG['THRESHOLDS']['one_class_svm']
)
plot_anomalies(data, oc_svm_anomalies, 'one_class_svm')
generate_report(data, oc_svm_anomalies, 'one_class_svm')
evaluate_detection(data['label'], oc_svm_anomalies, 'One-Class SVM')

# Autoencoder
autoencoder_anomalies, autoencoder_model, ae_threshold = detect_anomalies_autoencoder(
    data_scaled,
    threshold_ratio=CONFIG['THRESHOLDS']['autoencoder'],
    epochs=100,
    batch_size=64
)
plot_anomalies(data, autoencoder_anomalies, 'autoencoder')
generate_report(data, autoencoder_anomalies, 'autoencoder')
evaluate_detection(data['label'], autoencoder_anomalies, 'Autoencoder')


# Save models
iso_forest_model_path = os.path.join(CONFIG['MODEL_DIR'], 'isolation_forest_model.pkl')
oc_svm_model_path = os.path.join(CONFIG['MODEL_DIR'], 'one_class_svm_model.pkl')
autoencoder_model_path = os.path.join(CONFIG['MODEL_DIR'], 'autoencoder_model.h5')

pd.to_pickle(iso_forest_model, iso_forest_model_path)
pd.to_pickle(oc_svm_model, oc_svm_model_path)
autoencoder_model.save(autoencoder_model_path)

logger.info(f"Models saved to {CONFIG['MODEL_DIR']} directory.")

if __name__ == "__main__":
    main()