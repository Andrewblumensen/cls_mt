# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 12:21:58 2024

@author: hujo8
"""

import os
import pandas as pd
import numpy as np
# Initialize an empty list to store dataframes
dfs = []
df2s = []

# Path to the folder containing all subfolders
folder_path = 'runs/predict-cls'

# Loop through each subfolder
for folder_name in os.listdir(folder_path):
    folder_dir = os.path.join(folder_path, folder_name)
    
    # Check if the item in the folder_path is a directory
    if os.path.isdir(folder_dir):
        csv_path = os.path.join(folder_dir, 'class.csv')
        prop_path = os.path.join(folder_dir, 'prop.csv')
        
        # Check if class.csv exists in the current folder
        if os.path.exists(csv_path):
            # Read the CSV file into a dataframe
            df = pd.read_csv(csv_path, header=None, delimiter=';')
            
            # Add a 'label' column with the folder name
            df['label'] = folder_name
            
            # Append the dataframe to the list
            dfs.append(df)
            
        if os.path.exists(prop_path):
            # Read the CSV file into a dataframe
            df2 = pd.read_csv(prop_path, header=None, delimiter=';')
            
            # Append the dataframe to the list
            df2s.append(df2)

# Concatenate all dataframes into one big dataframe
big_df = pd.concat(dfs, ignore_index=True)

# Concatenate all dataframes into one big dataframe
big_df2 = pd.concat(df2s, ignore_index=True)


big_df.columns = ["1.","2.","3.","4.","5.","6.","7.","8.","9.","10.","label"]


# Assuming your dataframe has a column 'label' for ground truth and a column 'prediction' for predicted classes

# Calculate top-1 accuracy
top_1_accuracy = (big_df['label'] == big_df['1.']).mean()

# Print the result
print("Top-1 Accuracy:", top_1_accuracy)


# Calculate top-5 accuracy
def top_5_accuracy(row):
    return row['label'] in row[['1.', '2.', '3.', '4.', '5.']].values

# Apply the function row-wise and take the mean
top_5_accuracy = big_df.apply(top_5_accuracy, axis=1).mean()

# Print the result
print("Top-5 Accuracy:", top_5_accuracy)



from sklearn.metrics import confusion_matrix

# Create the confusion matrix
conf_matrix = confusion_matrix(big_df['label'], big_df['1.'])

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)



#%%

# Initialize an empty dictionary to store probabilities
probabilities_dict = {}

# Iterate through each row in big_df and big_df2 simultaneously
for index, (classes, probabilities) in enumerate(zip(big_df.values, big_df2.values)):
    # Iterate through each class and its probability
    for class_name, probability in zip(classes, probabilities):
        # If the class name is not in the dictionary, create an empty list
        if class_name not in probabilities_dict:
            probabilities_dict[class_name] = []
        # Append the probability to the corresponding class
        probabilities_dict[class_name].append(float(probability))

# Create a new DataFrame from the probabilities dictionary
smx1 = pd.DataFrame(probabilities_dict)
columns_list = smx1.columns.tolist()

smx = np.array(smx1)

labels = np.array(big_df['label'])

encoded_labels = np.array([columns_list.index(label) for label in labels])


#%%

# Problem setup
n = 1500  # number of calibration points
alpha = 0.1  # 1-alpha is the desired coverage


# Split the softmax scores into calibration and validation sets (save the shuffling)
idx = np.array([1] * n + [0] * (smx.shape[0] - n)) > 0
np.random.shuffle(idx)
cal_smx, val_smx = smx[idx, :], smx[~idx, :]
cal_labels, val_labels = encoded_labels[idx], encoded_labels[~idx]



"""
from sklearn.model_selection import train_test_split

# Problem setup
n = 1962  # number of calibration points
alpha = 0.1  # 1-alpha is the desired coverage

# Split the data into calibration and validation sets
cal_smx, val_smx, cal_labels, val_labels = train_test_split(smx, labels, test_size=0.5, random_state=42)
"""

#%%

# Get scores. calib_X.shape[0] == calib_Y.shape[0] == n
cal_pi = cal_smx.argsort(1)[:, ::-1]
cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[range(n), cal_labels]


# Get the score quantile
qhat = np.quantile(
    cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher"
)
# Deploy (output=list of length n, each element is tensor of classes)
val_pi = val_smx.argsort(1)[:, ::-1]
val_srt = np.take_along_axis(val_smx, val_pi, axis=1).cumsum(axis=1)
prediction_sets = np.take_along_axis(val_srt <= qhat, val_pi.argsort(axis=1), axis=1)



# Calculate empirical coverage
empirical_coverage = prediction_sets[
    np.arange(prediction_sets.shape[0]), val_labels
].mean()
print(f"The empirical coverage is: {empirical_coverage}")



# Calculate the lengths of all prediction sets
prediction_set_lengths = np.sum(prediction_sets, axis=1)

# Calculate the average length of all prediction sets
average_prediction_set_length = np.mean(prediction_set_lengths)

# Print the average length of prediction sets
print("Average length of prediction sets:", average_prediction_set_length)


#%%

import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy.optimize import brentq
from scipy.stats import binom
import seaborn as sns

# Problem setup
alpha = 0.2 # 1-alpha is the desired selective accuracy
delta = 0.2 # delta is the failure rate
lambdas = np.linspace(0,1,5000)

cal_yhats = np.argmax(cal_smx,axis=1); val_yhats = np.argmax(val_smx,axis=1)
cal_phats = cal_smx.max(axis=1); val_phats = np.max(val_smx,axis=1)


# Define selective risk
def selective_risk(lam): return (cal_yhats[cal_phats >= lam] != cal_labels[cal_phats >= lam]).sum()/(cal_phats >= lam).sum()
def nlambda(lam): return (cal_phats > lam).sum()
lambdas = np.array([lam for lam in lambdas if nlambda(lam) >= 30]) # Make sure there's some data in the top bin.
def invert_for_ub(r,lam): return binom.cdf(selective_risk(lam)*nlambda(lam),nlambda(lam),r)-delta
# Construct upper boud
def selective_risk_ub(lam): return brentq(invert_for_ub,0,0.9999,args=(lam,))
# Scan to choose lamabda hat
for lhat in np.flip(lambdas):
    if selective_risk_ub(lhat-1/lambdas.shape[0]) > alpha: break
# Deploy procedure on test data
predictions_kept = val_phats >= lhat


#%%

# Get indices of confident predictions
confident_indices = np.where(predictions_kept)[0]

# Get indices of inconfident predictions
inconfident_indices = np.where(~predictions_kept)[0]

# Get indices of correct predictions
correct_indices = np.where(val_yhats == val_labels)[0]

# Get indices of wrong predictions
wrong_indices = np.where(val_yhats != val_labels)[0]

# Count the occurrences
confident_correct = np.intersect1d(confident_indices, correct_indices).size
inconfident_wrong = np.intersect1d(inconfident_indices, wrong_indices).size
confident_wrong = np.intersect1d(confident_indices, wrong_indices).size
inconfident_correct = np.intersect1d(inconfident_indices, correct_indices).size

print("Confident and correct:", confident_correct)
print("Inconfident and wrong:", inconfident_wrong)
print("Confident and wrong:", confident_wrong)
print("Inconfident and correct:", inconfident_correct)


import seaborn as sns
import matplotlib.pyplot as plt

# Define the confusion matrix
confusion_matrix_values = np.array([[confident_correct,inconfident_correct],
                                    [confident_wrong, inconfident_wrong]])

# Define the labels for rows and columns
row_labels = ['Actual Correct', 'Actual Wrong']
column_labels = ['Predicted Correct', 'Predicted Wrong']

# Create a DataFrame for the confusion matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix_values, 
                                   index=row_labels,
                                   columns=column_labels)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_df, annot=True, fmt='d', cmap='Blues')
plt.title('Selective Classification Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.show()


#%%


# Calculate empirical selective accuracy
empirical_selective_accuracy = (val_yhats[predictions_kept] == val_labels[predictions_kept]).mean()
print(f"The empirical selective accuracy is: {empirical_selective_accuracy}")
false_flags = (val_yhats[~predictions_kept] == val_labels[~predictions_kept]).mean()
print(f"The fraction of false flags is: {false_flags}")
fraction_kept = predictions_kept.mean()
print(f"The fraction of data points kept is: {fraction_kept}")










