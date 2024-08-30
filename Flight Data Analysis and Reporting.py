import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

# Step 1: Simulate Data Collection
def simulate_flight_data(num_records=1000):
    np.random.seed(42)

    time = np.arange(0, num_records)
    altitude = np.random.normal(50000, 1000, num_records) # Simulated altitude (ft)
    velocity = np.random.normal(3000, 100, num_records) # Simulated velocity (mph)
    temperature = np.random.normal(-70, 5, num_records) # Simulated temperature (Celsius)

    data = pd.DataFrame({
        'Time (s)': time,
        'Altitude (ft)': altitude,
        'Velocity (mph)': velocity,
        'Temperature (C)': temperature
    })

    return data

# Step 2: Perform Data Analysis
def analyze_flight_data(data):
    analysis = {
        'Mean Altitude (ft)': data['Altitude (ft)'].mean(),
        'Max Altitude (ft)': data['Altitude (ft)'].max(),
        'Min Altitude (ft)': data['Altitude (ft)'].min(),
        'Mean Velocity (mph)': data['Velocity (mph)'].mean(),
        'Max Velocity (mph)': data['Velocity (mph)'].max(),
        'Min Velocity (mph)': data['Velocity (mph)'].min(),
        'Mean Temperature (C)': data['Temperature (C)'].mean(),
        'Max Temperature (C)': data['Temperature (C)'].max(),
        'Min Temperature (C)': data['Temperature (C)'].min(),
    }
    
    return pd.DataFrame(analysis, index=[0])  # Corrected line

# Step 3: Generate Report with Visualizations
def generate_report(data, analysis):
    sns.set(style="whitegrid")

    # Create visualizations
    plt.figure(figsize=(14, 10))

    plt.subplot(3, 1, 1,)
    sns.lineplot(x='Time (s)', y='Altitude (ft)', data=data, color='blue')
    plt.title('Altitude Over Time')

    plt.subplot(3, 1, 2)
    sns.lineplot(x='Time (s)', y='Velocity (mph)', data=data, color='green')
    plt.title('Velocity Over Time')

    plt.subplot(3, 1, 3)
    sns.lineplot(x='Time (s)', y='Temperature (C)', data=data, color='red')
    plt.title('Temperature Over Time')

    plt.tight_layout()
    plt.savefig('flight_data_visualizations.png')
    plt.close()

    # Generate PDF report
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Flight Data Analysis Report", ln=True, align='C')

    pdf.ln(10)

    for key, value in analysis.items():
        pdf.cell(200, 10, txt=f"{key}: {value:.2f}", ln=True)

    pdf.ln(10)

    pdf.cell(200, 10, txt="See attached visualizations for more details.", ln=True)

    pdf.image('flight_data_visualizations.png', x=10, y=70, w=180)

    pdf.output("flight_data_analysis_report.pdf")
    print("Report generated: flight_data_analysis_report.pdf")

# Main Execution
if __name__ == "__main__":
    # Step 1: Simulate flight data collection
    flight_data = simulate_flight_data()
    
    # Step 2: Analyze the flight data
    analysis_results = analyze_flight_data(flight_data)
    
    # Step 3: Generate the report with visualizations
    generate_report(flight_data, analysis_results.iloc[0].to_dict())  # Corrected line

