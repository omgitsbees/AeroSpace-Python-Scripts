import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Step 1: Simulate Propulsion System Data
def simulate_propulsion_data(num_samples=1000):
    np.random.seed(42)
    time = np.arange(num_samples)
    
    thrust = np.random.normal(loc=1000, scale=50, size=num_samples)
    fuel_consumption = np.random.normal(loc=200, scale=10, size=num_samples)
    temperature = np.random.normal(loc=800, scale=30, size=num_samples)
    vibration = np.random.normal(loc=5, scale=0.5, size=num_samples)
    
    data = {
        'Time': time,
        'Thrust': thrust,
        'Fuel Consumption': fuel_consumption,
        'Temperature': temperature,
        'Vibration': vibration
    }
    
    return pd.DataFrame(data)

propulsion_data = simulate_propulsion_data()

# Step 2: Analyze Propulsion System Performance
def analyze_performance(df):
    analysis = {
        'Average Thrust (N)': df['Thrust'].mean(),
        'Total Fuel Consumption (kg)': df['Fuel Consumption'].sum(),
        'Max Temperature (°C)': df['Temperature'].max(),
        'Max Vibration (g)': df['Vibration'].max(),
    }
    
    # Detect anomalies using Z-score
    df['Thrust Z-Score'] = zscore(df['Thrust'])
    df['Temperature Z-Score'] = zscore(df['Temperature'])
    df['Vibration Z-Score'] = zscore(df['Vibration'])
    
    anomalies = df[(df['Thrust Z-Score'].abs() > 3) |
                   (df['Temperature Z-Score'].abs() > 3) |
                   (df['Vibration Z-Score'].abs() > 3)]
    
    return pd.DataFrame([analysis]), anomalies

performance_summary, anomalies_detected = analyze_performance(propulsion_data)

# Step 3: Generate Alerts for Anomalies
def generate_alerts(anomalies):
    if not anomalies.empty:
        print("Alert: Anomalies detected in propulsion system performance!")
        print(anomalies)
    else:
        print("No anomalies detected.")

generate_alerts(anomalies_detected)

# Step 4: Generate Performance Report
def generate_report(performance_summary, anomalies, file_name='propulsion_system_report.txt'):
    with open(file_name, 'w') as report_file:
        report_file.write("Propulsion System Performance Report\n")
        report_file.write("="*40 + "\n\n")
        
        report_file.write("Performance Summary:\n")
        report_file.write(performance_summary.to_string(index=False) + "\n\n")
        
        if not anomalies.empty:
            report_file.write("Anomalies Detected:\n")
            report_file.write(anomalies.to_string(index=False) + "\n\n")
        else:
            report_file.write("No anomalies detected.\n\n")
        
        report_file.write("End of Report.\n")

generate_report(performance_summary, anomalies_detected)

# Step 5: Visualization (Optional)
def visualize_data(df):
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 2, 1)
    sns.lineplot(x='Time', y='Thrust', data=df)
    plt.title('Thrust Over Time')
    
    plt.subplot(2, 2, 2)
    sns.lineplot(x='Time', y='Fuel Consumption', data=df)
    plt.title('Fuel Consumption Over Time')
    
    plt.subplot(2, 2, 3)
    sns.lineplot(x='Time', y='Temperature', data=df)
    plt.title('Temperature Over Time')
    
    plt.subplot(2, 2, 4)
    sns.lineplot(x='Time', y='Vibration', data=df)
    plt.title('Vibration Over Time')
    
    plt.tight_layout()
    plt.show()

visualize_data(propulsion_data)
