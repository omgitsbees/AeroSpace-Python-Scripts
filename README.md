NASA Dataset.py = This grabs data from a publcily available .csv file from NASA's datasets, titled Outgassing_Db_20240828.csv

-----------------------------------------------------------------------------------------------------------------------------------

best_prices_new_glenn.py = this simulates parts, and prices as if they are on the Blue Origin New Glenn rocket, grabbing prices from multiple vendors in order to compare 

-----------------------------------------------------------------------------------------------------------------------------------

cad_parts_list.py = this automation script takes the parts list from a CAD file and creates a csv with the weight, and size dimensions of each part 

-----------------------------------------------------------------------------------------------------------------------------------

Flight Data Analysis and Reporting = This script will simulate the collection of flight data, perform statistical analysis, and generate a report with visualizations.

-----------------------------------------------------------------------------------------------------------------------------------

The simulate_flight_data() function generates random data for altitude, velocity, and temperature. This mimics real flight data collection. The analyze_flight_data() function calculates basic statistics such as mean, max, and min values for each metric. The generate_report() function creates a PDF report. It includes a summary of the analysis and visualizes the data over time.

-----------------------------------------------------------------------------------------------------------------------------------

Supply Chain Optimization = automate the analysis and reporting of vendor performance, parts availability, and cost-effectiveness.

Simulate Data Collection: 'simulate_supply_chain_data() generates random data for vendors, parts, costs, lead times, and quality scores.

Data Analysis: 'analyze_supply_chain_data() calculates mean cost, lead time, and quality score for each vendor.

Optimization: 'optimize_supplier_selection() uses linear programming to minimize the total cost while ensuring each part is supplied and no vendor is overused.

Report Generation: 'generate_report() creates a PDF report with vendor performance summary and the results of the supplier optimization.

-----------------------------------------------------------------------------------------------------------------------------------

Propulsion System Performance Monitoring.py = The goal of this project is to develop a Python script that monitors key performance indicators (KPIs) of a propulsion system, such as thrust, fuel consumption, temperature, and vibration levels, and generates alerts or reports when certain thresholds are exceeded.

Data Ingestion: The script will simulate the ingestion of propulsion system data from various sensors.

Data Processing: It will process the data to calculate performance metrics like efficiency and detect anomalies.

Alerting: The script will generate alerts when performance metrics exceed predefined thresholds.

Reporting: It will generate a summary report of the propulsion system's performance over a specific period.

Simulate Propulsion Data: The simulate_propulsion_data function generates synthetic data for thrust, fuel consumption, temperature, and vibration levels over time.

Analyze Performance: The analyze_performance function computes key performance metrics and detects anomalies using the Z-score method.

Generate Alerts: If any anomalies are detected, an alert is printed to the console with details of the anomalies.

Generate Report: The generate_report function writes a detailed performance report to a text file, including summaries and any detected anomalies.

Visualization: The optional visualization step plots the time series data for better insight into the propulsion system's performance.

-----------------------------------------------------------------------------------------------------------------------------------

