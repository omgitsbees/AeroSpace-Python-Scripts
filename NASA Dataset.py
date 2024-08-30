import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os

# Setup logging
logging.basicConfig(filename='etl_process.log', level=logging.INFO)

def load_and_process_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path} with {len(df)} rows at {datetime.now()}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def analyze_data(df):
    try:
        # Convert TML to numeric, forcing any errors to NaN
        df['TML'] = pd.to_numeric(df['TML'], errors='coerce')
        
        # Drop rows with NaN values in TML
        df = df.dropna(subset=['TML'])
        
        # Filter materials with low TML (e.g., TML < 1)
        low_tml_materials = df[df['TML'] < 1]
        
        # Sort by TML
        low_tml_materials_sorted = low_tml_materials.sort_values(by='TML')
        
        logging.info(f"Data analysis complete with {len(low_tml_materials_sorted)} rows at {datetime.now()}")
        return low_tml_materials_sorted
    except Exception as e:
        logging.error(f"Error during data analysis: {e}")
        raise

def plot_results(df, plot_file_path):
    try:
        top_10_materials = df.head(10)
        if not top_10_materials.empty:
            plt.figure(figsize=(10, 6))
            plt.barh(top_10_materials['Sample Material'], top_10_materials['TML'], color='skyblue')
            plt.xlabel('Total Mass Loss (TML)')
            plt.ylabel('Sample Material')
            plt.title('Top 10 Materials with Lowest Total Mass Loss (TML)')
            plt.gca().invert_yaxis()  # Invert y-axis to show the lowest TML at the top
            plt.savefig(plot_file_path)
            plt.show()
            logging.info(f"Plot saved to {plot_file_path} at {datetime.now()}")
        else:
            logging.warning("No data available to plot.")
    except Exception as e:
        logging.error(f"Error creating plot: {e}")
        raise

def send_email(subject, body, attachments, to_email):
    try:
        msg = MIMEMultipart()
        msg['From'] = 'youremail@example.com'  # Replace with your actual email
        msg['To'] = to_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        for file in attachments:
            attachment = open(file, 'rb')
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f"attachment; filename= {os.path.basename(file)}")
            msg.attach(part)

        server = smtplib.SMTP('smtp.gmail.com', 587)  # Using Gmail's SMTP server
        server.starttls()
        server.login('youremail@gmail.com', 'yourpassword')  # Replace with your credentials
        text = msg.as_string()
        server.sendmail('youremail@gmail.com', to_email, text)
        server.quit()
        logging.info(f"Email sent to {to_email} at {datetime.now()}")
    except Exception as e:
        logging.error(f"Error sending email: {e}")
        raise

def main():
    file_path = r'C:\Users\kyleh\Downloads\Outgassing_Db_20240828.csv'
    output_file_path = r'C:\Users\kyleh\Downloads\Low_TML_Materials.csv'
    plot_file_path = r'C:\Users\kyleh\Downloads\TML_Plot.png'
    to_email = 'kyleheimbigner@gmail.com'

    df = load_and_process_data(file_path)
    low_tml_materials_sorted = analyze_data(df)
    
    # Save the filtered dataset to a new CSV file
    low_tml_materials_sorted.to_csv(output_file_path, index=False)
    
    # Plot and save the results
    plot_results(low_tml_materials_sorted, plot_file_path)
    
    # Send email with the results
    send_email(
        subject="ETL Process Complete - Results and Plot",
        body="Please find the attached results and plot from the ETL process.",
        attachments=[output_file_path, plot_file_path],
        to_email=to_email
    )

if __name__ == "__main__":
    main()
