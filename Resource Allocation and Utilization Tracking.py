import tkinter as tk
from tkinter import ttk, messagebox
import pulp
import schedule
import time
import threading
import bcrypt
import pandas as pd
import statsmodels.api as sm
from jira import JIRA


# Global variables
resources = ['Engineer 1', 'Engineer 2', 'Technician 1', 'Technician 2']
projects = ['Project A', 'Project B', 'Project C']
utilization_data = pd.Series([80, 75, 90, 85, 70])  # Example utilization data for forecasting

# Function to perform AI-based resource allocation
def optimal_resource_allocation(resources, projects):
    prob = pulp.LpProblem("Resource Allocation", pulp.LpMaximize)
    allocation_vars = {(i, j): pulp.LpVariable(f"assign_{i}_{j}", 0, 1, pulp.LpBinary)
                       for i in resources for j in projects}
    prob += pulp.lpSum([allocation_vars[i, j] for i in resources for j in projects])
    for i in resources:
        prob += pulp.lpSum([allocation_vars[i, j] for j in projects]) <= 1
    prob.solve()
    allocation = {}
    for i in resources:
        for j in projects:
            if pulp.value(allocation_vars[i, j]) == 1:
                allocation[i] = j
    return allocation


# Function to simulate resource utilization over time
def simulate_time_step():
    print("Simulating resource usage...")
    # Simulate resource consumption, reduce available resources


# Simulation thread to run in background
def run_simulation():
    while True:
        schedule.run_pending()
        time.sleep(1)


# Setup a simulation mode that runs in the background
def start_simulation():
    schedule.every(1).seconds.do(simulate_time_step)
    threading.Thread(target=run_simulation, daemon=True).start()


# Function to authenticate user (bcrypt hashing)
def hash_password(plain_password):
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(plain_password.encode(), salt)


def check_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode(), hashed_password)


# Example Jira Integration
def fetch_jira_issues():
    try:
        jira_options = {'server': 'https://your-jira-domain.atlassian.net'}
        jira = JIRA(options=jira_options, basic_auth=('email@example.com', 'api_token'))
        issues = jira.search_issues('project=PROJ')
        return [(issue.key, issue.fields.summary) for issue in issues]
    except Exception as e:
        messagebox.showerror("Jira Error", f"Error fetching Jira issues: {e}")
        return []


# Forecast resource utilization
def forecast_utilization(data):
    model = sm.tsa.statespace.SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()
    forecast = results.get_forecast(steps=12)
    return forecast.predicted_mean


# Tkinter UI Setup
class ResourceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Resource Allocation and Utilization Tracking")
        self.create_ui()

    def create_ui(self):
        # Create the main interface
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.grid(row=0, column=0, padx=10, pady=10)

        # User login form
        self.username_label = ttk.Label(self.main_frame, text="Username:")
        self.username_label.grid(row=0, column=0)
        self.username_entry = ttk.Entry(self.main_frame)
        self.username_entry.grid(row=0, column=1)

        self.password_label = ttk.Label(self.main_frame, text="Password:")
        self.password_label.grid(row=1, column=0)
        self.password_entry = ttk.Entry(self.main_frame, show='*')
        self.password_entry.grid(row=1, column=1)

        self.login_button = ttk.Button(self.main_frame, text="Login", command=self.login)
        self.login_button.grid(row=2, column=1)

        # AI-based allocation button
        self.allocate_button = ttk.Button(self.main_frame, text="Optimize Allocation", command=self.optimize_allocation)
        self.allocate_button.grid(row=3, column=0)

        # Simulate usage button
        self.simulate_button = ttk.Button(self.main_frame, text="Start Simulation", command=start_simulation)
        self.simulate_button.grid(row=3, column=1)

        # Forecast utilization button
        self.forecast_button = ttk.Button(self.main_frame, text="Forecast Utilization", command=self.show_forecast)
        self.forecast_button.grid(row=4, column=0)

        # Jira integration button
        self.jira_button = ttk.Button(self.main_frame, text="Fetch Jira Issues", command=self.show_jira_issues)
        self.jira_button.grid(row=4, column=1)

        # Output text box
        self.output_text = tk.Text(self.main_frame, height=10, width=50)
        self.output_text.grid(row=5, column=0, columnspan=2)

    def login(self):
        # Sample hashed password for demo purposes
        stored_hashed_password = hash_password("password123")
        username = self.username_entry.get()
        password = self.password_entry.get()

        if check_password(password, stored_hashed_password):
            messagebox.showinfo("Login Success", "Login successful!")
        else:
            messagebox.showerror("Login Failed", "Invalid username or password")

    def optimize_allocation(self):
        allocation = optimal_resource_allocation(resources, projects)
        output = "\n".join([f"{resource} -> {allocation[resource]}" for resource in allocation])
        self.output_text.insert(tk.END, f"Optimal Allocation:\n{output}\n")

    def show_forecast(self):
        forecast = forecast_utilization(utilization_data)
        self.output_text.insert(tk.END, f"Forecasted Utilization:\n{forecast}\n")

    def show_jira_issues(self):
        issues = fetch_jira_issues()
        output = "\n".join([f"{issue[0]}: {issue[1]}" for issue in issues])
        self.output_text.insert(tk.END, f"Jira Issues:\n{output}\n")


# Main entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = ResourceApp(root)
    root.mainloop()