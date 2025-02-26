import csv

# Example data with a corrected file path
data = [
    {'soil_name': 'Alluvial Soil', 'image_path': r'C:\Users\dhany\AppData\Local\Temp\0bfb5f73-f92a-4f1d-b99c-983a7fa6b585_archive.zip.585\Dataset\test\Alluvial soil\1000_F_240425429_YL91trtDxXQl8L0OKP7zyngeSb63olAC.jpg'},
    
]

# File path for the CSV
csv_file_path = 'soil_data.csv'

# Writing data to CSV
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['soil_name', 'image_path'])
    
    # Write the header
    writer.writeheader()
    
    # Write the rows
    for row in data:
        writer.writerow(row)

print(f"CSV file '{csv_file_path}' created successfully!")
