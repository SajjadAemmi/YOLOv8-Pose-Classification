import os
import csv
import json


directory_path = './fitness_poses_csvs_out'

all_data = {}

# Loop through all CSV files in the directory
for file_name in os.listdir(directory_path):
    if file_name.endswith('.csv'):
        class_name = file_name[:-(len('.csv'))]
        
        with open(os.path.join(directory_path, file_name), 'r') as csvfile:
            # Read the CSV file and convert its contents to a dictionary
            csv_data = list(csv.reader(csvfile))
            # Add the dictionary to the list of all data

            for row in csv_data:
                row[1:] = map(float, row[1:])

            all_data[class_name] = csv_data

# Convert the list of dictionaries to a JSON string
json_data = json.dumps(all_data)

# Set the path for the output JSON file
output_file_path = './js_version/pose_samples.json'

# Write the JSON string to the output file
with open(output_file_path, 'w') as output_file:
    output_file.write(json_data)
