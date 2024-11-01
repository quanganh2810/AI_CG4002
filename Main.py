import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from PytorchModel import MLP
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pickle
from scipy.stats import kurtosis

torch.manual_seed(35)
torch.set_printoptions(linewidth=200, precision=4, threshold=10000)

folder_path = "C:\\Users\\Nguyen Quang Choach\\Desktop\\Witcher\\1nov_glove_imu_data_andrew"
dataFiles = glob.glob(os.path.join(folder_path, "*.csv"))
#Data reading and processing - remove redundant columns, remove rows with '0', split the data into frame and calculate necessary aggregates
#Function to aggregate the data by splitting into chunks and calculate min, max, mean and std
def dataAggregator(df):
    results = []
    num_action = 25
    for label, group in df.groupby('Label'):
        chunk_size = len(group) // num_action
        for i in range(num_action): 
            chunk = group.iloc[i * chunk_size: (i + 1) * chunk_size]
            chunk_stats = {}
            for col in group.columns[:-1]:  # Skip 'label' column
                chunk_stats[f'{col}Min'] = chunk[col].min()
                chunk_stats[f'{col}Max'] = chunk[col].max()
                chunk_stats[f'{col}Mean'] = chunk[col].mean()
                chunk_stats[f'{col}Std'] = chunk[col].std()
                #chunk_stats[f'{col}Range'] = chunk[col].max() - chunk[col].min()
            chunk_stats['Label'] = label
            results.append(chunk_stats)
    dfResults = pd.DataFrame(results)
    return dfResults

def dataFilter(rawDataDf):
    filtered_rows = []
    i = 0
    num = 0

    # Loop through DataFrame rows
    while i < len(rawDataDf):
        # Find the next row where the acceleration magnitude exceeds 1.5
        while i < len(rawDataDf):
            accel_magnitude = np.sqrt(
                rawDataDf.loc[i, 'Accel X']**2 + 
                rawDataDf.loc[i, 'Accel Y']**2 + 
                rawDataDf.loc[i, 'Accel Z']**2
            )
            if accel_magnitude > 1.5:
                num += 1
                print(f"THRESHOLD EXCEEDED at {i}. Number of exceeds = {num}")
                break
            i += 1
        
        # If the loop completes without finding more rows, end the function
        if i >= len(rawDataDf):
            break

        # Append the next 70 rows from this point to filtered_rows
        print(f"APPENDING {i} to {i + 60}")
        filtered_rows.extend(rawDataDf.iloc[i:i + 60].values)

        # Move the index forward by 100 (70 collected + 30 discarded)
        i += 100

    # Convert the collected rows back to a DataFrame with the original columns
    filtered_df = pd.DataFrame(filtered_rows, columns=rawDataDf.columns)
    return filtered_df

dataDf = []
save_path = "C:\\Users\\Nguyen Quang Choach\\Desktop\\Witcher\\Filtered Data"
for file in dataFiles:
    if "glove" in file and "soccer" not in file:
        data = pd.read_csv(file, delimiter=',', header=0)
        label = data.loc[0, "Label"]
        csv_file_path = f"{save_path}/{label}.csv"
        #Filter the data 
        data = dataFilter(data)
        #print(data)
        data.to_csv(csv_file_path, index=False)
        #Aggregate the data
        data = dataAggregator(data)
        dataDf.append(data)
processedDataDf = pd.concat(dataDf, ignore_index=True)
#print(processedDataDf)
#Split the data 
trainDataDf, testDataDf = train_test_split(processedDataDf, test_size=0.2, random_state=42)
X_train = trainDataDf.drop(columns='Label').values
y_train = trainDataDf['Label'].values
X_test = testDataDf.drop(columns='Label').values
y_test = testDataDf['Label'].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
with open("C:\\Users\\Nguyen Quang Choach\\Desktop\\Witcher\\Orders\\All\\scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

#print(X_test)
X_test = scaler.transform(X_test)

print(f'Dimensions of X_test: {X_test.shape}')
num_classes = len(set(y_train))
print(f'Number of classes: {num_classes}')

df_X_test = pd.DataFrame(X_test, columns=[f'Feature_{i}' for i in range(X_test.shape[1])])
df_y_test = pd.DataFrame(y_test, columns=['label'])

# Save DataFrames to CSV files
df_X_test.to_csv('X.csv', index=False)
df_y_test.to_csv('y.csv', index=False)

mapping = {1: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6}

# Apply the mapping to y_train
y_train = np.vectorize(mapping.get)(y_train)
y_test = np.vectorize(mapping.get)(y_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
num_classes = len(set(y_train))  # Ensure this is the correct number of classes
print(f'Number of classes: {num_classes}')
print("Unique labels in y_train:", np.unique(y_train))
# Hyperparameters
print(X_train)
input_size = X_train.shape[1]
output_size = len(set(y_train))
print(f"input_size is {input_size}")
print(f"output_size is {output_size}")
model = MLP(input_size=input_size, output_size=output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

def print_model_parameters_to_file(model, file_path='param.txt'):
    with open(file_path, 'w') as f:
        for name, param in model.named_parameters():
            if param.requires_grad:
                f.write(f'{name}:\n')
                f.write(f'Shape: {param.shape}\n')  
                flattened_tensor = param.data.view(-1).tolist()
                f.write(f'{flattened_tensor}\n')

# Training the model
def train_model(model, criterion, optimizer, X_train, y_train, epochs=100, batch_size=32):

    for epoch in range(epochs):
        for i in range(0, X_train.size(0), batch_size):
            # Get mini-batch
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        # Print the loss every few epochs
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Function to evaluate the model and calculate accuracy
def evaluate_model(model, X_test, y_test):
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_test, predicted)
        
        # Calculate the confusion matrix
        #cm = confusion_matrix(y_test, predicted)
        
        # Plot confusion matrix
        #plt.figure(figsize=(10, 7))
        #sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        #plt.xlabel('Predicted labels')
        #plt.ylabel('True labels')
        #plt.title('Confusion Matrix')
        #plt.show()
    return accuracy

# Train the model
train_model(model, criterion, optimizer, X_train_tensor, y_train_tensor, epochs=50, batch_size=32)

# Evaluate the model
accuracy = evaluate_model(model, X_test_tensor, y_test_tensor)
print(f'Accuracy: {accuracy * 100:.2f}%')

unlabeled_data_path = "C:\\Users\\Nguyen Quang Choach\\Desktop\\Witcher\\IMU_debug_data.csv"
unlabeled_data = pd.read_csv(unlabeled_data_path)

# Function to aggregate unlabeled data into sections of 60 rows
def aggregate_unlabeled_data(df):
    results = []
    num_sections = len(df) // 60  # Calculate the number of 60-row sections
    
    for i in range(num_sections):
        section = df.iloc[i * 60:(i + 1) * 60]
        section_stats = {}
        for col in section.columns:  # Aggregate statistics for each column
            section_stats[f'{col}Min'] = section[col].min()
            section_stats[f'{col}Max'] = section[col].max()
            section_stats[f'{col}Mean'] = section[col].mean()
            section_stats[f'{col}Std'] = section[col].std()
            #section_stats[f'{col}Range'] = section[col].max() - section[col].min()
        results.append(section_stats)
    
    aggregated_df = pd.DataFrame(results)
    return aggregated_df

# Aggregate the unlabeled data
aggregated_unlabeled_data = aggregate_unlabeled_data(unlabeled_data)
print(aggregated_unlabeled_data)
aggregated_unlabeled_data = aggregated_unlabeled_data.values
print(aggregated_unlabeled_data)

with open("C:\\Users\\Nguyen Quang Choach\\Desktop\\Witcher\\Orders\\All\\scaler.pkl", "rb") as file:
    loaded_scaler = pickle.load(file)

# Standardize the data using the same scaler that was used on training data
aggregated_unlabeled_data_scaled = loaded_scaler.transform(aggregated_unlabeled_data)

# Convert the standardized data into a tensor for the model
X_unlabeled_tensor = torch.tensor(aggregated_unlabeled_data_scaled, dtype=torch.float32)

# Use the trained model to make predictions on the aggregated data
with torch.no_grad():
    model.eval()  # Set the model to evaluation mode
    predictions = model(X_unlabeled_tensor)
    _, predicted_labels = torch.max(predictions, 1)

# Output the predictions
print("Predicted Labels for Unlabeled Data:")
print(predicted_labels.numpy())