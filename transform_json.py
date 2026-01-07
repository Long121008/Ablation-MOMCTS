import json
import os

input_file = "logs/nsga2/bi_kp/test_100/v3/samples_1~300.json"
output_file = "logs/nsga2/bi_kp/test_100/v3/samples_1~300_trans.json"

try:
    # Load the old JSON file
    with open(input_file, "r") as infile:
        data = json.load(infile)

    # Transform the data
    transformed_data = []
    for key, value in data.items():
        if "100" in value and isinstance(value["100"], list):
            transformed_data.append({"score": value["100"]})

    # Save the transformed data to a new JSON file
    with open(output_file, "w") as outfile:
        json.dump(transformed_data, outfile, indent=4)

    print(f"Transformation complete. Transformed data saved to {output_file}")

except Exception as e:
    print(f"An error occurred: {e}")

