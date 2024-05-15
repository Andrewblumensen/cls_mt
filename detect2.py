

import json

# Path to the JSON file
file_path = "runs/detect/exp7/6503.json"

# Function to read JSON file and count colonies
def count_colonies(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)
        predictions = data["predictions"]
        num_colonies = len(predictions)
        return num_colonies

# Call the function and print the result
num_colonies = count_colonies(file_path)
print("Number of colonies found:", num_colonies)


