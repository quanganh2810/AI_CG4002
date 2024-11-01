import pandas as pd
import glob
import os

folder_path = "C:\\Users\\Nguyen Quang Choach\\Desktop\\Witcher\\haoyang_data_set_24oct"  # Replace with your folder path
save_folder_path = "C:\\Users\\Nguyen Quang Choach\\Desktop\\Witcher\\Merged Data"  # Replace with your folder path
dataFiles = glob.glob(os.path.join(folder_path, "*.csv"))

# Create a dictionary to group files by sport and number
file_groups = {}

# Group files by sport and number (e.g., basketball_1)
for file in dataFiles:
    file_name = os.path.basename(file)
    base_name = "_".join(file_name.split("_")[:2])  # Get the sport and number part
    if base_name not in file_groups:
        file_groups[base_name] = {}
    if "glove" in file_name:
        file_groups[base_name]["glove"] = file
    elif "ankle" in file_name:
        file_groups[base_name]["ankle"] = file

# Process each group
merged_data = []
for base_name, files in file_groups.items():
    if "glove" in files and "ankle" in files:
        # Load glove and ankle files
        glove_data = pd.read_csv(files["glove"], delimiter=',', header=0)
        ankle_data = pd.read_csv(files["ankle"], delimiter=',', header=0)

        # Compare lengths and truncate the longer file from the front
        if len(glove_data) > len(ankle_data):
            glove_data = glove_data.iloc[-len(ankle_data):].reset_index(drop=True)
        elif len(ankle_data) > len(glove_data):
            ankle_data = ankle_data.iloc[-len(glove_data):].reset_index(drop=True)

        # Drop 'label' column from the ankle data only
        if 'Label' in glove_data.columns:
            glove_data = glove_data.drop(columns=['Label'])

        # Add suffixes to columns, excluding 'label' column in glove_data
        # glove_data = glove_data.add_suffix("_glove")
        ankle_data = ankle_data.add_suffix("_ankle").rename(columns={"Label_ankle": "Label"})

        # Merge horizontally
        merged = pd.concat([glove_data, ankle_data], axis=1)
        merged_data.append(merged)

        # Optionally save the merged data to a file
        save_path = f"{save_folder_path}/{base_name}_merged.csv"
        merged.to_csv(save_path, index=False)
        print(f"Merged file saved as {save_path}")
