import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import StandardScaler

# Load the CSV files
folder_path = "C:\\Users\\Nguyen Quang Choach\\Desktop\\Witcher\\1nov_glove_imu_data_andrew"
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

for file_path in csv_files:
    if "glove" in file_path: #and "soccer" not in file_path:
        data = pd.read_csv(file_path)
    
    # Calculate the Net Acceleration
        data['Net Accel'] = np.sqrt(data['Accel X']**2 + data['Accel Y']**2 + data['Accel Z']**2)
    
    # Define the time axis based on the sampling rate (assuming 1 unit per frame)
        time = [i * 1 for i in range(len(data))]
    
    # List of columns to plot
        columns_to_plot = ['Net Accel']
    
    # Plot each column as a line graph
        for column in columns_to_plot:
            plt.figure(figsize=(10, 4))
            plt.plot(time, data[column], label=column)
            plt.title(f'{file_path} Over Time')
            plt.xlabel('Frames')
            plt.ylabel(column)
            plt.legend()
            plt.grid(True)
            # plt.figure(figsize=(10, 4))
            # plt.scatter(time, data[column], label=column, s=10)  # 's' sets the dot size
            # plt.title(f'{file_path} Over Time')
            # plt.xlabel('Frames')
            # plt.ylabel(column)
            # plt.legend()
            # plt.grid(True)
        # Extract the unique labels (assuming label is categorical and exists in the dataset)
            unique_labels = data['Label'].unique() if 'Label' in data.columns else 'No labels'
        
        # Display the labels at the bottom of the graph
            plt.text(0.5, -0.15, f'Labels: {unique_labels}', 
                 horizontalalignment='center', verticalalignment='center', 
                 transform=plt.gca().transAxes, fontsize=12)
        
            plt.tight_layout()
            plt.show()
