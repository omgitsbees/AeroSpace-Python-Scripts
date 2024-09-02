import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from fpdf import FPDF
import tkinter as tk
from tkinter import filedialog, messagebox
import geopandas as gpd
from shapely.geometry import Point, Polygon

class EnvironmentalImpactAnalysis:
    def __init__(self):
        self.data = None
        self.results = {}

    def load_data(self, file_path):
        """Load data from CSV or Excel file."""
        if file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            self.data = pd.read_excel(file_path)
        print("Data loaded successfully.")

    def preprocess_data(self):
        """Handle missing values and normalize data."""
        self.data.fillna(self.data.mean(), inplace=True)
        print("Data preprocessed.")

    def calculate_emissions(self):
        """Calculate CO2 emissions."""
        # Sample calculation
        self.data['CO2_Emissions'] = self.data['Fuel_Consumed'] * 3.16  # CO2 per kg of fuel
        self.results['emissions'] = self.data['CO2_Emissions'].sum()
        print("Emissions calculated.")

    def calculate_noise_pollution(self):
        """Estimate noise levels and impact radius."""
        # Placeholder implementation
        self.data['Noise_Level'] = self.data['Distance'] * 0.05  # Example formula
        self.results['noise'] = self.data['Noise_Level'].mean()
        print("Noise pollution estimated.")

    def assess_ecological_footprint(self):
        """Analyze ecological impact such as land use and biodiversity loss."""
        # Placeholder implementation
        self.data['Ecological_Footprint'] = self.data['Area_Used'] * 1.2  # Example formula
        self.results['ecological'] = self.data['Ecological_Footprint'].sum()
        print("Ecological footprint assessed.")

    def predictive_modeling(self):
        """Run predictive models to forecast future impacts."""
        X = self.data[['Fuel_Consumed', 'Distance']]
        y = self.data['CO2_Emissions']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        self.results['mse'] = mse
        print(f"Predictive modeling completed. MSE: {mse}")

    def visualize_data(self):
        """Visualize data with plots."""
        sns.lineplot(data=self.data, x='Mission', y='CO2_Emissions')
        plt.title("CO2 Emissions by Mission")
        plt.show()

    def interactive_maps(self):
        """Generate interactive maps to visualize impact zones."""
        # Placeholder for geopandas visualization
        gdf = gpd.GeoDataFrame(self.data, geometry=gpd.points_from_xy(self.data.Longitude, self.data.Latitude))
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        base = world.plot(color='white', edgecolor='black')
        gdf.plot(ax=base, marker='o', color='red', markersize=5)
        plt.title("Impact Zones")
        plt.show()

    def generate_report(self, file_name):
        """Generate a PDF report."""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Environmental Impact Analysis Report", ln=True, align="C")
        pdf.cell(200, 10, txt=f"Total CO2 Emissions: {self.results['emissions']} kg", ln=True, align="L")
        pdf.cell(200, 10, txt=f"Average Noise Level: {self.results['noise']} dB", ln=True, align="L")
        pdf.cell(200, 10, txt=f"Total Ecological Footprint: {self.results['ecological']} hectares", ln=True, align="L")
        pdf.cell(200, 10, txt=f"Predictive Model MSE: {self.results['mse']}", ln=True, align="L")
        pdf.output(file_name)
        print(f"Report {file_name} generated.")

    def build_gui(self):
        """Build a basic GUI for the tool using tkinter."""
        def load_file():
            file_path = filedialog.askopenfilename()
            self.load_data(file_path)

        def run_analysis():
            self.preprocess_data()
            self.calculate_emissions()
            self.calculate_noise_pollution()
            self.assess_ecological_footprint()
            self.predictive_modeling()
            self.visualize_data()
            self.generate_report("Environmental_Impact_Report.pdf")
            messagebox.showinfo("Analysis Complete", "Environmental Impact Analysis completed successfully!")

        root = tk.Tk()
        root.title("Environmental Impact Analysis Tool")

        load_button = tk.Button(root, text="Load Data", command=load_file)
        load_button.pack()

        analyze_button = tk.Button(root, text="Run Analysis", command=run_analysis)
        analyze_button.pack()

        root.mainloop()

# Example Usage
if __name__ == "__main__":
    analysis = EnvironmentalImpactAnalysis()
    analysis.build_gui()  # Launch the GUI for user interaction
