import numpy as np
import pandas as pd
import sqlite3
import requests
import json
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns
import pdfkit
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
import logging
import tkinter as tk
from tkinter import messagebox
from cryptography.fernet import Fernet

# Logging setup
logging.basicConfig(filename='risk_assessment.log', level=logging.INFO)

# Security (Encryption) Setup
key = Fernet.generate_key()
cipher_suite = Fernet(key)

class RiskAssessment:

    def __init__(self, project_name):
        self.project_name = project_name
        self.conn = sqlite3.connect(f'{project_name}_risk.db')
        self.cursor = self.conn.cursor()
        self.setup_database()

    def setup_database(self):
        # Create tables for storing risk data
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS risks (
                               id INTEGER PRIMARY KEY,
                               description TEXT,
                               probability REAL,
                               impact REAL,
                               risk_level REAL,
                               date_assessed TEXT)''')
        self.conn.commit()

    def add_risk(self, description, probability, impact):
        # Calculate risk level
        risk_level = probability * impact
        date_assessed = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        encrypted_description = cipher_suite.encrypt(description.encode())
        self.cursor.execute('''INSERT INTO risks (description, probability, impact, risk_level, date_assessed)
                               VALUES (?, ?, ?, ?, ?)''',
                            (encrypted_description, probability, impact, risk_level, date_assessed))
        self.conn.commit()
        logging.info(f"Added risk: {description}, Risk Level: {risk_level}")

    def generate_report(self):
        # Generate a report of all risks
        self.cursor.execute('''SELECT * FROM risks''')
        risks = self.cursor.fetchall()
        decrypted_risks = [(r[0], cipher_suite.decrypt(r[1]).decode(), r[2], r[3], r[4], r[5]) for r in risks]
        df = pd.DataFrame(decrypted_risks, columns=['ID', 'Description', 'Probability', 'Impact', 'Risk Level', 'Date Assessed'])
        report_html = df.to_html()
        with open('risk_report.html', 'w') as f:
            f.write(report_html)
        pdfkit.from_file('risk_report.html', 'risk_report.pdf')
        logging.info("Generated risk report.")

    def send_alert(self, to_email):
        # Send email alerts if risk exceeds a threshold
        self.cursor.execute('''SELECT * FROM risks WHERE risk_level > 0.7''')
        high_risks = self.cursor.fetchall()
        if high_risks:
            msg = MIMEMultipart()
            msg['From'] = 'risk_assessment@aerospace.com'
            msg['To'] = to_email
            msg['Subject'] = "High Risk Alert"
            body = "The following risks have been identified as high:\n\n"
            for risk in high_risks:
                decrypted_description = cipher_suite.decrypt(risk[1]).decode()
                body += f"Description: {decrypted_description}, Risk Level: {risk[4]}\n"
            msg.attach(MIMEText(body, 'plain'))
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login('your_email@gmail.com', 'your_password')
            server.send_message(msg)
            server.quit()
            logging.info("Sent high-risk alert email.")

    def historical_data_analysis(self):
        # Analyze historical data
        self.cursor.execute('''SELECT * FROM risks''')
        risks = self.cursor.fetchall()
        decrypted_risks = [(r[0], cipher_suite.decrypt(r[1]).decode(), r[2], r[3], r[4], r[5]) for r in risks]
        df = pd.DataFrame(decrypted_risks, columns=['ID', 'Description', 'Probability', 'Impact', 'Risk Level', 'Date Assessed'])
        sns.histplot(df['Risk Level'])
        plt.title('Risk Level Distribution')
        plt.show()

    def machine_learning_risk_prediction(self):
        # Predict risk using advanced machine learning models
        self.cursor.execute('''SELECT * FROM risks''')
        risks = self.cursor.fetchall()
        decrypted_risks = [(r[0], cipher_suite.decrypt(r[1]).decode(), r[2], r[3], r[4], r[5]) for r in risks]
        df = pd.DataFrame(decrypted_risks, columns=['ID', 'Description', 'Probability', 'Impact', 'Risk Level', 'Date Assessed'])
        X = df[['Probability', 'Impact']]
        y = (df['Risk Level'] > 0.7).astype(int)  # 1 if high risk, 0 otherwise

        # Standardize the data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Risk prediction model accuracy: {accuracy}")

    def get_real_time_data(self):
        # Example: Real-time weather data from OpenWeatherMap API
        api_key = "your_openweathermap_api_key"
        location = "Seattle"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}"
        response = requests.get(url)
        data = json.loads(response.text)
        if response.status_code == 200:
            weather_description = data['weather'][0]['description']
            temp = data['main']['temp'] - 273.15  # Convert from Kelvin to Celsius
            wind_speed = data['wind']['speed']
            logging.info(f"Real-time weather data: {weather_description}, Temp: {temp}C, Wind Speed: {wind_speed}m/s")
            return weather_description, temp, wind_speed
        else:
            logging.error("Failed to retrieve real-time weather data.")
            return None, None, None

    def add_real_time_data(self):
        weather_description, temp, wind_speed = self.get_real_time_data()
        if weather_description:
            self.add_risk(f"Weather Risk: {weather_description}", temp / 50, wind_speed / 15)

    def close(self):
        # Close the database connection
        self.conn.close()
        logging.info("Closed the database connection.")

# GUI for Risk Assessment
class RiskAssessmentApp(tk.Tk):
    def __init__(self, risk_assessment):
        super().__init__()
        self.title("Automated Risk Assessment")
        self.geometry("500x400")
        self.risk_assessment = risk_assessment

        # Widgets
        self.label = tk.Label(self, text="Add New Risk")
        self.label.pack(pady=10)

        self.desc_label = tk.Label(self, text="Risk Description")
        self.desc_label.pack(pady=5)
        self.desc_entry = tk.Entry(self, width=50)
        self.desc_entry.pack(pady=5)

        self.prob_label = tk.Label(self, text="Probability (0 to 1)")
        self.prob_label.pack(pady=5)
        self.prob_entry = tk.Entry(self, width=50)
        self.prob_entry.pack(pady=5)

        self.impact_label = tk.Label(self, text="Impact (0 to 1)")
        self.impact_label.pack(pady=5)
        self.impact_entry = tk.Entry(self, width=50)
        self.impact_entry.pack(pady=5)

        self.add_button = tk.Button(self, text="Add Risk", command=self.add_risk)
        self.add_button.pack(pady=10)

        self.report_button = tk.Button(self, text="Generate Report", command=self.generate_report)
        self.report_button.pack(pady=10)

        self.alert_button = tk.Button(self, text="Send Alert", command=self.send_alert)
        self.alert_button.pack(pady=10)

        self.analysis_button = tk.Button(self, text="Historical Data Analysis", command=self.historical_data_analysis)
        self.analysis_button.pack(pady=10)

        self.ml_button = tk.Button(self, text="Run Machine Learning Prediction", command=self.machine_learning_risk_prediction)
        self.ml_button.pack(pady=10)

        self.rt_button = tk.Button(self, text="Add Real-Time Data", command=self.add_real_time_data)
        self.rt_button.pack(pady=10)

    def add_risk(self):
        description = self.desc_entry.get()
        probability = float(self.prob_entry.get())
        impact = float(self.impact_entry.get())
        self.risk_assessment.add_risk(description, probability, impact)
        messagebox.showinfo("Success", "Risk added successfully!")

    def generate_report(self):
        self.risk_assessment.generate_report()
        messagebox.showinfo("Success", "Risk report generated successfully!")

    def send_alert(self):
        email = tk.simpledialog.askstring("Input", "Enter recipient email address:")
        if email:
            self.risk_assessment.send_alert(email)
            messagebox.showinfo("Success", "Alert sent successfully!")
    
    def historical_data_analysis(self):
        self.risk_assessment.historical_data_analysis()

    def machine_learning_risk_prediction(self):
        self.risk_assessment.machine_learning_risk_prediction()
        messagebox.showinfo("Success", "Machine Learning Prediction run successfully!")
    
    def add_real_time_data(self):
        self.risk_assessment.add_real_time_data()
        messagebox.showinfo("Success", "Real-time data added successfully!")

    def on_closing(self):
        self.risk_assessment.close()
        self.destroy()

# Main function
def main():
    risk_assessment = RiskAssessment("Aerospace_Project")
    app = RiskAssessmentApp(risk_assessment)
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

if __name__ == "__main__":
    main()