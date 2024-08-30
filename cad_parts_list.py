from pyautocad import Autocad, APoint

# Initialize AutoCAD connection (create a new instance if none exists)
try:
    acad = Autocad(create_if_not_exists=True)
    print("AutoCAD connected.")
except Exception as e:
    print("Failed to connect to AutoCAD:", e)
    exit()

# List to hold extracted part data
part_data = []

# Iterate through objects in the AutoCAD model space
for obj in acad.iter_objects():
    if obj.ObjectName == "AcDb3dSolid":  # Assuming 3D solids represent parts
        part_name = obj.Name
        weight = obj.MassProp()[0]  # Assuming MassProp returns a tuple with weight as the first element
        dimensions = f"{obj.BoundingBox[1].x - obj.BoundingBox[0].x}x{obj.BoundingBox[1].y - obj.BoundingBox[0].y}x{obj.BoundingBox[1].z - obj.BoundingBox[0].z}"
        
        # Add the data to the list
        part_data.append({"Part": part_name, "Weight (kg)": weight, "Dimensions (m)": dimensions})

# Convert the list to a DataFrame
df = pd.DataFrame(part_data)

# Save the DataFrame to a CSV file
df.to_csv("cad_parts_list.csv", index=False)
print("\nCAD parts list saved to 'cad_parts_list.csv'")
