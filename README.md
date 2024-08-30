NASA Dataset.py = This grabs data from a publcily available .csv file from NASA's datasets, titled Outgassing_Db_20240828.csv

best_prices_new_glenn.py = this simulates parts, and prices as if they are on the Blue Origin New Glenn rocket, grabbing prices from multiple vendors in order to compare 

cad_parts_list.py = this automation script takes the parts list from a CAD file and creates a csv with the weight, and size dimensions of each part 

Flight Data Analysis and Reporting = This script will simulate the collection of flight data, perform statistical analysis, and generate a report with visualizations.
The simulate_flight_data() function generates random data for altitude, velocity, and temperature. This mimics real flight data collection. The analyze_flight_data() function calculates basic statistics such as mean, max, and min values for each metric. The generate_report() function creates a PDF report. It includes a summary of the analysis and visualizes the data over time.

Supply Chain Optimization = automate the analysis and reporting of vendor performance, parts availability, and cost-effectiveness. Here's a step-by-step guide on how you could structure such a project:
Simulate Data Collection: 'simulate_supply_chain_data() generates random data for vendors, parts, costs, lead times, and quality scores.
Data Analysis: 'analyze_supply_chain_data() calculates mean cost, lead time, and quality score for each vendor.
Optimization: 'optimize_supplier_selection() uses linear programming to minimize the total cost while ensuring each part is supplied and no vendor is overused.
Report Generation: 'generate_report() creates a PDF report with vendor performance summary and the results of the supplier optimization.
