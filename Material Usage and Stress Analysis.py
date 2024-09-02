import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import filedialog
from mpl_toolkits.mplot3d import Axes3D

# Function to load material data
def load_material_data():
    file_path = filedialog.askopenfilename(title="Select Material Data File", filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
    if file_path.endswitch('.csv'):
        material_data = pd.read_csv(file_path)
    else:
        material_data = pd.read_excel(file_path)
    return material_data

# Function to calculate materail usage efficiency
def calculate_material_usage(component_volume, material_density):
    material_usage = component_volume * material_density
    return material_usage

# Function to calculate safety factor
def calculate_safety_factor(applied_stress, yield_strength):
    safety_factor = yield_strength / applied_stress
    return safety_factor

# Function for Finite Element Analysis (simplified)
def finite_element_analysis(component, material_properties):
    # Placeholder: Assume uniform stress distribution for simplicity
    stress_distribution = np.random.uniform(0.5, 1.5, size=component.shape) * material_properties['Yield_Strength']
    return stress_distribution

# Function to optimize material usage
def optimize_material_usage(component, material_properties):
    def objective(material_usage):
        return material_usage  # Minimize material usage
    
    def constraint1(material_properties):
        return calculate_safety_factor(applied_stress=component['Stress'], yield_strength=material_properties['Yield_Strength']) - 1  # Ensure safety factor >= 1
    
    constraints = [{'type': 'ineq', 'fun': constraint1}]
    result = minimize(objective, material_properties['Density'], constraints=constraints)
    return result

# Function for predictive modeling
def predictive_modeling(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    mse = mean_square_error(y_test, predictions)
    return model, mse

# Function to generate a report
def generate_report(material_data, component_data, stress_analysis, optimization_results):
    with pdfPages('material_stress_analysis_report.pdf') as pdf:
        plt.figure(figsize=(8, 6))
        plt.title("Material Usage and Stress Distribution")
        plt.hist(stress_analysis_flatten(), bins=30, alpha=0.7, label='Stress Distribution')
        plt.axyline(x=np.mean(stress_analysis), color='r', linestyle='--', label='Mean Stress')
        plt.xlabel("Stress (Pa)")
        plt.ylabel("Frequency")
        plt.legend()
        pdf.savefig()
        plt.close()

        # More detailed plots and tables can be added here

        # Generate a summary table
        summary_table = pd.DataFrame({
            'Component': component_data['Name'],
            'Material': component_data['Material'],
            'Volume': component_data['Volume'],
            'Stress': stress_analysis.mean(axis=1),
            'Safety Factor': component_data['Safety_Factor']
        })
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('tight')
        ax.axis('off')
        the_table = ax.table(cellText=summary_table.values, colLabels=summary_table.columns, cellLoc='center', loc='center')
        pdf.savefig()
        plt.close()

# Interactive GUI
def gui():
    root = tk.Tk()
    root.title("Material Usage and Stress Analysis")

    def run_analysis():
        material_data = load_material_data()
        component_volume = float(volume_entry.get())
        material_density = float(density_entry.get())

        material_usage = calculate_material_usage(component_volume, material_density)
        material_label.config(text=f"Material Usage: {material_usage:.2f} kg")

        # Dummy component data for FEA
        component = pd.DataFrame({
            'Name': ['Component1', 'Component2'],
            'Volume': [component_volume, component_volume],
            'Material': ['Material1', 'Material2'],
            'Stress': [200, 250]
        })

        stress_analysis = finite_element_analysis(component, material_data)
        safety_factors = Calculate_safety_factor(stress_analysis, material_data['Yield_Strength'].iloc[0])

        Optimization_results = optimize_material_usage(component, material_data.iloc[0])

        # Display results and generate report
        generate_report(material_data, component, stress_analysis, optimization_results)
        result_label.config(text="Analysis complete. Report generated.")

    # GUI layout
    tk.Label(root, text="Component Volume (m^3)").grid(row=0)
    tk.Label(root, text="Material Density (kg/m^3)").grid(row=1)
    volume_entry = tk.Entry(root)
    density_entry = tk.Entry(root)
    volume_entry.grid(row=0, column=1)
    density_entry.grid(row=1, column=1)

    tk.Button(root, text='Run Analysis', command=run_analysis).grid(row=3, column=0, pady=4)
    material_label = tk.Label(root, text="")
    material_label.grid(row=4, column=0, columnspan=2)
    result_label = tk.Label(root, text="")
    result_label.grid(row=5, column=0, columnspan=2)

    root.mainloop()

if __name__ == "__main__":
    gui()
