import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from fpdf import FPDF

# Step 1: Simulate Data Collection
def simulate_supply_chain_data(num_vendors=5, num_parts=10):
    np.random.seed(42)
    
    vendors = [f'Vendor_{i+1}' for i in range(num_vendors)]
    parts = [f'Part_{i+1}' for i in range(num_parts)]
    
    data = {
        'Part': np.random.choice(parts, num_vendors * num_parts),
        'Vendor': np.random.choice(vendors, num_vendors * num_parts),  # Corrected typo
        'Cost': np.random.uniform(100, 1000, num_vendors * num_parts),  # Cost per part
        'LeadTime': np.random.uniform(1, 30, num_vendors * num_parts),   # Lead time in days
        'QualityScore': np.random.uniform(0, 100, num_vendors * num_parts)  # Quality score (0 to 100)
    }
    
    df = pd.DataFrame(data)
    return df

# The rest of the script remains the same...

# Step 2: Perform Data Analysis
def analyze_supply_chain_data(df):
    summary = df.groupby('Vendor').agg({
        'Cost': 'mean',
        'LeadTime': 'mean',
        'QualityScore': 'mean'
    }).reset_index()
    
    return summary

# Step 3: Optimization
def optimize_supplier_selection(df):
    # Aggregate cost data to handle duplicates
    df_agg = df.groupby(['Vendor', 'Part']).agg({'Cost': 'mean'}).reset_index()
    
    # Create a pivot table for cost
    cost_matrix = df_agg.pivot(index='Vendor', columns='Part', values='Cost').fillna(0)
    
    # Define the coefficients for the cost minimization problem
    c = cost_matrix.values.flatten()
    
    # Constraints for each part to be supplied exactly once
    num_parts = len(cost_matrix.columns)
    A_eq = np.zeros((num_parts, len(c)))
    b_eq = np.ones(num_parts)
    
    for i in range(num_parts):
        A_eq[i, i::num_parts] = 1
    
    # Constraints for each vendor (e.g., no more than 1 part per vendor)
    num_vendors = len(cost_matrix.index)
    A_ub = np.zeros((num_vendors, len(c)))
    b_ub = np.ones(num_vendors)
    
    for i in range(num_vendors):
        A_ub[i, i*num_parts:(i+1)*num_parts] = 1
    
    bounds = [(0, 1) for _ in range(len(c))]
    
    # Linear programming to minimize cost
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    return result

# Step 4: Generate Report
def generate_report(summary, optimization_result):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Supply Chain Optimization Report", ln=True, align='C')
    
    pdf.ln(10)
    pdf.cell(200, 10, txt="Vendor Performance Summary:", ln=True)
    
    for index, row in summary.iterrows():
        pdf.cell(200, 10, txt=f"Vendor: {row['Vendor']}", ln=True)
        pdf.cell(200, 10, txt=f"  Average Cost: ${row['Cost']:.2f}", ln=True)
        pdf.cell(200, 10, txt=f"  Average Lead Time: {row['LeadTime']:.2f} days", ln=True)
        pdf.cell(200, 10, txt=f"  Average Quality Score: {row['QualityScore']:.2f}", ln=True)
    
    pdf.ln(10)
    pdf.cell(200, 10, txt="Optimized Supplier Selection:", ln=True)
    
    if optimization_result.success:
        pdf.cell(200, 10, txt=f"Optimization Result: Cost = ${optimization_result.fun:.2f}", ln=True)
        pdf.cell(200, 10, txt="Supplier Selection:", ln=True)
        for i, value in enumerate(optimization_result.x):
            if value > 0:
                vendor_index = i // len(cost_matrix.columns)
                part_index = i % len(cost_matrix.columns)
                pdf.cell(200, 10, txt=f"  Vendor: {cost_matrix.index[vendor_index]}, Part: {cost_matrix.columns[part_index]}", ln=True)
    else:
        pdf.cell(200, 10, txt="Optimization failed. Please check the constraints and data.", ln=True)
    
    pdf.output("supply_chain_optimization_report.pdf")
    print("Report generated: supply_chain_optimization_report.pdf")

# Main Execution
if __name__ == "__main__":
    # Step 1: Simulate supply chain data
    supply_chain_data = simulate_supply_chain_data()
    
    # Step 2: Analyze the supply chain data
    performance_summary = analyze_supply_chain_data(supply_chain_data)
    
    # Step 3: Optimize supplier selection
    optimization_result = optimize_supplier_selection(supply_chain_data)
    
    # Step 4: Generate the report
    generate_report(performance_summary, optimization_result)
