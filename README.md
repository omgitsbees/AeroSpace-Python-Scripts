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

Launch Schedule Optimization = The idea behind this project is to optimize the scheduling of rocket launches, taking into consideration factors like weather conditions, rocket availability, payload readiness, crew availability, and air traffic.

step-by-step breakdown:
Step 1: Define the Problem

We want to optimize the scheduling of rocket launches. The inputs to the problem might include:

    Launch Windows: Available time slots for launching rockets.
    Rocket Availability: Whether a rocket is ready for launch.
    Payload Readiness: Whether the payload is ready for launch.
    Weather Conditions: Whether the weather is favorable for launch.
    Crew Availability: Whether the crew is available and ready.

Step 2: Define Constraints

Some constraints to consider:

    Rockets cannot be launched outside of their designated launch windows.
    Weather conditions must be favorable for a launch to occur.
    Payload must be ready and loaded.
    Crew must be available and rested.

Step 3: Implement the Optimization

simple optimization technique where we will try to assign the best possible launch windows based on the constraints.

Constraint Satisfaction Problem (CSP): treat this as a CSP where each launch needs to satisfy several constraints (e.g., weather, rocket readiness, payload, etc.).

Optimization Algorithm: use an optimization algorithm like Genetic Algorithms (GA) to find the optimal launch schedule.

Data Structures: use sophisticated data structures like classes to represent rockets, payloads, and schedules.

Concurrency: Introduce concurrency with asyncio to simulate real-time checks and decision-making.

Logging and Error Handling: Add logging and exception handling for robustness.

Visualization: Generate a Gantt chart to visualize the schedule.

Explanation of the Enhanced Code:

    Classes and Data Structures: The Rocket, Payload, Weather, Crew, LaunchWindow, and Schedule are more clearly defined using namedtuples and classes for better data management.

    Constraint Satisfaction and Genetic Algorithm: The code uses a Genetic Algorithm (GA) for optimization, which is a more sophisticated method than simple loops and checks. The GA simulates a natural selection process to find an optimal solution by generating, mutating, and crossing over schedules.

    Concurrency: asyncio is used to simulate real-time decision-making and to allow the script to handle multiple tasks concurrently, which can be especially useful if the scheduling process needs to be responsive to real-time changes.

    Logging and Error Handling: Logging is implemented to track the progress and decisions made during the optimization process, which adds professionalism to the script and helps in debugging and analysis.

    Visualization: A Gantt chart is generated using matplotlib to visually represent the launch schedule, which is a critical feature for demonstrating the results to stakeholders.

-----------------------------------------------------------------------------------------------------------------------------------

Anomaly Detection in Telemetry Data = comprehensive Python script that performs anomaly detection in telemetry data using a combination of statistical methods, machine learning, and visualization. This script is designed for automation, allowing it to process telemetry data, detect anomalies, and generate detailed reports.

comprehensive anomaly detection system specifically designed for aerospace telemetry data. This system will incorporate the following features:

    Synthetic Data Generation: Since real aerospace telemetry data might not be readily available, we'll simulate realistic telemetry data including parameters like altitude, velocity, acceleration, temperature, pressure, and fuel levels.

    Data Preprocessing: Handling missing values, scaling, and smoothing the data to prepare it for analysis.

    Anomaly Detection Techniques:
        Statistical Methods: Z-Score and Interquartile Range (IQR) methods.
        Machine Learning Methods:
            Isolation Forest
            One-Class SVM
            Autoencoders (Deep Learning): Specifically suited for capturing complex patterns in time-series data.

    Visualization: Plotting the telemetry data with highlighted anomalies for easy interpretation.

    Reporting: Generating detailed reports summarizing detected anomalies, including timestamps and parameter specifics.

    Modularity and Scalability: The code will be organized into functions and classes for easy maintenance and scalability.

    Real-time Processing Capability: Designing the system to handle streaming data for real-time anomaly detection.

Technologies and Libraries Used:

    Python 3.x
    NumPy and Pandas: Data manipulation and analysis.
    Matplotlib and Seaborn: Data visualization.
    Scikit-Learn: Machine learning algorithms.
    TensorFlow/Keras: Building and training deep learning models (Autoencoders).
    PyOD: Specialized library for outlier detection.
    Streamlit (Optional): For creating an interactive dashboard/interface.

    Detailed Explanation:
1. Synthetic Data Generation:

Since real aerospace telemetry data may not be publicly available, we generate synthetic data that mimics real-world scenarios.

    Features Generated:
        Altitude: Simulated around 10,000 meters with some variance.
        Velocity: Around 250 m/s.
        Acceleration: Centered around 0 m/s² to represent steady flight with occasional fluctuations.
        Temperature: Around 15°C with variations.
        Pressure: Around 1013 hPa (standard atmospheric pressure).
        Fuel Level: Starting with 5000 liters and varying slightly.

    Anomaly Injection:
        We introduce anomalies by randomly altering the values of these features significantly at random timestamps.
        This simulates scenarios like sudden drops or spikes in readings due to malfunctions or external factors.

2. Data Preprocessing:

    Handling Missing Values: Although our synthetic data doesn't have missing values, this step ensures robustness by forward and backward filling any gaps.
    Scaling: We use StandardScaler to normalize the data, which is crucial for many anomaly detection algorithms to perform effectively.

3. Anomaly Detection Methods:

We implement multiple methods to detect anomalies, each with its strengths:
a. Z-Score Method:

    Concept: Identifies how many standard deviations an observation is from the mean.
    Threshold: Typically set at 3, meaning any data point beyond 3 standard deviations is considered an anomaly.

b. Interquartile Range (IQR) Method:

    Concept: Uses the middle 50% of data to identify outliers.
    Threshold: Commonly set at 1.5 times the IQR above the 75th percentile or below the 25th percentile.

c. Isolation Forest:

    Concept: An ensemble method that isolates anomalies by randomly selecting a feature and splitting value.
    Advantages: Efficient on large datasets and can handle high-dimensional data.
    Parameters:
        Contamination: Proportion of anomalies in the data.

d. One-Class SVM:

    Concept: Learns the boundary of normal data and identifies points outside this boundary as anomalies.
    Kernel: Radial Basis Function (RBF) kernel captures non-linear relationships.
    Parameters:
        Nu: An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.

e. Autoencoder (Deep Learning):

    Concept: Neural network trained to reconstruct its input. High reconstruction error indicates anomalies.
    Architecture:
        Encoder: Compresses input data.
        Decoder: Reconstructs data from the compressed representation.
    Training:
        Trained on normal data to minimize reconstruction error.
    Detection:
        Reconstruction error is calculated, and a threshold is set to identify anomalies.

4. Visualization:

    For each detection method, we plot the time series of each feature and highlight detected anomalies.
    Plots are saved in the specified directory for review.

5. Reporting:

    Detailed reports are generated in CSV format listing all detected anomalies with their timestamps and feature values.
    These reports are essential for post-analysis and validation.

6. Evaluation:

    We use classification metrics like precision, recall, F1-score, and confusion matrix to evaluate the performance of each detection method against the known anomalies.
    This helps in understanding which method performs best under different scenarios.

7. Model Saving:

    Trained models are saved for future use or deployment.
    Models like Isolation Forest and One-Class SVM are saved using pickle, while the Autoencoder model is saved in HDF5 format.

8. Modularity and Extensibility:

    The code is organized into functions, making it easy to maintain and extend.
    New detection methods or features can be added seamlessly.

9. Logging:

    Comprehensive logging is implemented to track the execution flow and debug if necessary.
    Logs provide timestamps and detailed messages for each major step.

10. Real-time Processing Capability:

    While the current implementation processes batch data, it can be extended to handle streaming data by adapting the functions to process incoming data points or windows
