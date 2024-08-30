import pandas as pd
import random

# Define the parts of the New Glenn rocket
parts = [ 
    "Engine", "Fuel Tank", "Oxidizer Tank", "Avionics System", "Guidance System",
    "Fairing", "Heat Shield", "Thrust Structure", "Landing Legs", "Interstage",
    "Payload Adapter", "Pressurization System", "Propellant Lines", "Telemetry System"    
]

# Define the vendors
vendors = ["Vendor A", "Vendor B", "Vendor C"]

# Generate mock prices for each part from each vendor
data = []
for part in parts:
    for vendor in vendors:
        price = round(random.uniform(100000, 5000000), 2) # Prices between $100k and $5M
        data.append({"Part": part, "Vendor": vendor, "Price": price})

# Create a DataFrame
df = pd.DataFrame(data)

# Group by part and find the vendor with the lowest price for each part
best_prices = df.loc[df.groupby("Part")["Price"].idxmin()]

# Display the results
print("\nBest Prices by Part:")
print(best_prices)

# Save the results to a CSV file
best_prices.to_csv("best_prices_new_glenn.csv", index=False)
print("\nBest prices saved to 'best_prices_new_glenn.csv'")