#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# CustomerInsights.py

"""
Description:
Author: Kyle Heckard
Date Created: October 21, 2024
Version 1.0
Python Version: 3.12.3
Dependencies: pandas, matplotlib, scipy, seaborn, scikit-learn, numpy, imblearn
"""

# %% Import libraries

import pandas as pd
import warnings as warn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.calibration import CalibratedClassifierCV
from datetime import datetime
from imblearn.combine import SMOTETomek
import random as rand

# %% Load Raw Data (csv)

clientData_raw = pd.read_csv("Clients.csv")
appointmentData_raw = pd.read_csv("Appointments.csv")

print('-data imported')

# %% General Declarations

col_index = None #Holds column index
ds = None        #Holds currently worked on data slice
df = None

# %% Filter out clients with no appointment data and reset index

clientData_filtered = clientData_raw[clientData_raw['Account_Number']
                                     .isin(appointmentData_raw['Account_Number'])]
clientData_filtered = clientData_filtered.reset_index(drop=True)

# Make copy of filtered data to preprocess
clientData = clientData = clientData_filtered.copy()

print('--client data filtered')

# %% ignore SettingWithCopyWarning

warn.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warn.filterwarnings("ignore", category=UserWarning, module="sklearn")

# %% rename columns

# Rename 'Sex' to 'c_Sex' so that I can reuse 'Sex'
clientData.rename(columns={'Sex': 'c_Sex'}, inplace=True)

# Rename 'Zip_PostalCode' to 'Zip' to shorten name or ease of use
clientData.rename(columns={'Zip_PostalCode': 'Zip'}, inplace=True)

# Rename 'Number_of_Appointments' to 'Num_Appts' to shorten length
clientData.rename(columns={'Number_of_Appointments': 'Num_Appts'},
                  inplace=True)

# %% Replace colomn 'c_Sex' with new column 'Sex' that uses integers to 
#    represent client gender

# Rename 'Sex' to 'c_Sex' so that I can reuse 'Sex'
clientData.rename(columns={'Sex': 'c_Sex'}, inplace=True) # rename

# If sex is M then 1, if F then 2, else nan
ds = clientData['c_Sex']
clientData.loc[:, 'Sex'] = np.where(ds == 'M', 1, #Male
                           np.where(ds == 'F', 2, #Female
                           np.where(ds == 'O', 3, #Other
                           np.nan)))              #No Entry

# Get index of c_Sex and move Sex behind it
col_index = clientData.columns.get_loc('c_Sex')
clientData.insert(col_index + 1, 'Sex', clientData.pop('Sex'))

# %% Add column 'Age' to clientData and calculate 'Age' from Birth_Year

# Get current year
current_year = datetime.now().year

# Where Birth_Year is 1900, age is nan, else age = currentYear - BirthYear
clientData.loc[:, 'Age'] = np.where(clientData['Birth_Year'] == 1900,
                           np.nan,
                           current_year - clientData['Birth_Year'])

# Get index of BirthYear and move 'Age' behind it
col_index = clientData.columns.get_loc('Birth_Year')
clientData.insert(col_index + 1, 'Age', clientData.pop('Age'))

# %% Convert Martial Status to Integers

# Convert Material Statuses to integers, else nan
ds = clientData['Marital_Status']
clientData.loc[:, 'Marital'] = np.where(ds == 'M', 1, #Married
                               np.where(ds == 'C', 2, #Common-Law
                               np.where(ds == 'D', 3, #Divorced
                               np.where(ds == 'W', 4, #Widowed
                               np.where(ds == 'S', 5, #Single
                               np.where(ds == 'O', 6, #Other
                               np.nan))))))           #No Entry

# Get index of 'Marital_Status' and move 'Marital' behind it
col_index = clientData.columns.get_loc('Marital_Status')
clientData.insert(col_index + 1, 'Marital', clientData.pop('Marital'))

# %% Convert Employment Status to Integers

# Convert Employment_Status to integers, else nan
ds = clientData['Employment_Status']
clientData.loc[:, 'Employment'] = np.where(ds == 'E', 1,  #Employed
                                  np.where(ds == 'S', 2,  #Student
                                  np.where(ds == 'R', 3,  #Retired
                                  np.where(ds == 'UE', 4, #Unemployed
                                  np.where(ds == 'O', 5,  #Other
                                  np.nan)))))             #No Entry

# Get index of 'Employment_Status' and move 'Employment' behind it
col_index = clientData.columns.get_loc('Employment_Status')
clientData.insert(col_index + 1, 'Employment', clientData.pop('Employment'))

# %% Convert 'Is condition due to an accident' to integers

# Convert 'Is_condtion_due_to_an_accident' to binary integers, else nan
ds = clientData['Is_condition_due_to_an_accident']
clientData.loc[:, 'Accident'] = np.where(ds == "[u'No']", 0,  #No
                                np.where(ds == 'No', 0,       #No
                                np.where(ds == "[u'Yes']", 1, #Yes
                                np.where(ds == 'Yes', 1,      #Yes
                                np.nan))))                    #No Entry

# Get index of 'Is_condition_due_to_an_accident' and move"Accident" behind it
col_index = clientData.columns.get_loc('Is_condition_due_to_an_accident')
clientData.insert(col_index + 1, 'Accident', clientData.pop('Accident'))

# %% Remove unused columns

del clientData['City']
del clientData['State_Province']
del clientData['Country']
del clientData['c_Sex']
del clientData['Birth_Year']
del clientData['Marital_Status']
del clientData['Employment_Status']
del clientData['Is_condition_due_to_an_accident']
del clientData['Reminder_Preference']
del clientData['Subscribed_Reminders']
del clientData['Subscribed_Recalls']
del clientData['Subscribed_Email_Campaigns']
del clientData['Subscribed_Birthday_Campaigns']
del clientData['Subscribed_Availability_Campaigns']
del clientData['Subscribe_Referral_Campaigns']
del clientData['Subscribe_Client_Satisfaction']

# %% Randomly assign values to NaNs based on Frequency of values in same column

# Fills NaNs based on how often other values in a column occur to maintain distribution
def fill_na_with_frequencies(series, value_counts):
    na_indices = series[series.isna()].index
    fill_values = np.random.choice(value_counts.index, size=len(na_indices), p=value_counts.values)
    series.loc[na_indices] = fill_values
    return series

# Zip Codes
ds = clientData['Zip']
value_counts = ds.value_counts(normalize=True)
ds = fill_na_with_frequencies(ds, value_counts)

# Sex
ds = clientData['Sex']
value_counts = ds.value_counts(normalize=True)
ds = fill_na_with_frequencies(ds, value_counts)

# Age
ds = clientData['Age']
value_counts = ds.value_counts(normalize=True)
ds = fill_na_with_frequencies(ds, value_counts)

# Marital_Status
ds = clientData['Marital']
value_counts = ds.value_counts(normalize=True)
ds = fill_na_with_frequencies(ds, value_counts)

# Employment_Status
ds = clientData['Employment']
value_counts = ds.value_counts(normalize=True)
ds = fill_na_with_frequencies(ds, value_counts)

# Is condition caused by accident
ds = clientData['Accident']
value_counts = ds.value_counts(normalize=True)
ds = fill_na_with_frequencies(ds, value_counts)

print('---client data prepared')

# %% Copy AppointmentData_raw

appointmentData = appointmentData_raw.copy()

# %% Rename columns

# Rename 'Service' to 'a_Service' so that 'Service' can be reused
appointmentData.rename(columns={'Service': 'a_Service'}, inplace=True)

# Rename 'Status' to 'a_Status' so that 'Status' can be reused
appointmentData.rename(columns={'Status': 'a_Status'}, inplace=True)

# %% Set appointmentData[Date] to Datetime format

appointmentData['Date'] = pd.to_datetime(appointmentData['Date'], format='%b %d, %Y @ %I:%M%p')

# %% Extract Pieces of DateTime object and add columns for them

# Extract Year, Quarter, Month, DayOfWeek, and Hour and add columns
appointmentData['Year'] = appointmentData['Date'].dt.year
appointmentData['Quarter'] = appointmentData['Date'].dt.quarter
appointmentData['Month'] = appointmentData['Date'].dt.month
appointmentData['Str_Day_of_Week'] = appointmentData['Date'].dt.day_name()
appointmentData['Hour'] = appointmentData['Date'].dt.hour

# convert Day_of_Week to Integer Format
ds = appointmentData['Str_Day_of_Week']
appointmentData.loc[:, 'Day_of_Week'] = np.where(ds == 'Sunday', 0,
                                        np.where(ds == 'Monday', 1,
                                        np.where(ds == 'Tuesday', 2,
                                        np.where(ds == 'Wednesday', 3,
                                        np.where(ds == 'Thursday', 4,
                                        np.where(ds == 'Friday', 5,
                                        np.where(ds == 'Saturday', 6,
                                        np.nan)))))))

# Get index of 'Str_Day_of_Week' and insert 'Day_of_Week' behind it
col_index = appointmentData.columns.get_loc('Str_Day_of_Week')
appointmentData.insert(col_index + 1, 'Day_of_Week', appointmentData.pop('Day_of_Week'))

# %% List Services to help convert to Integer format

services = ['30 minute 30 Minute Add-On',
            '60 minute Deep Tissue Massage',
            '60 minute Relaxation Massage',
            '60 minute Pregnancy Massage',
            '60 minute Insurance Massage',
            '60 minute Pregnancy Insurance Massage',
            '60 minute L&amp;I Massage',
            '60 minute Motor Vehicle Accident (MVA) Massage',
            '90 minute Deep Tissue Massage',
            '90 minute Relaxation Massage',
            '90 minute Insurance Massage',
            '90 minute Pregnancy Insurance Massage'
            ]

# %% Convert Services to integers

# Search for a value in array and return index
def find_index(value, array):
    try:
        return array.index(value)
    except ValueError:
        return -1

# Add column 'Service' and convert values to Integers base on index in
# services[] sharing same value
ds = appointmentData['a_Service']
appointmentData['Service'] = ds.apply(lambda x: find_index(x, services))

# %% Add column 'Length' with values based on length of services in minutes

# Search for value in array return integer based on index of value
def find_length(value, array):
    try:
        arg_30 = (array.index(value) == 0)
        arg_60 = ((array.index(value) >= 1) and (array.index(value) <= 7))
        arg_90 = ((array.index(value) >= 8) and (array.index(value) <= 11))
        
        if arg_30:
            return 30
        if arg_60:
            return 60
        if arg_90:
            return 90
    except ValueError:
        np.nan
   
# Add column 'Length' and assign each row an integer that reflect the length
# of that appointments service provided in minutes
ds = appointmentData['a_Service']
appointmentData['Length'] = ds.apply(lambda x: find_length(x, services))

# %% Add column 'Insurance' with integer based on if service is an
#    insurance massage

# Search for value in array and return binary Integer based on index of value
def find_insurance(value, array):
    try:
        arg_0_1 = ((array.index(value) >= 0) and (array.index(value) <= 3))
        arg_0_2 = ((array.index(value) >= 8) and (array.index(value) <= 9))
        arg_1_1 = ((array.index(value) >= 4) and (array.index(value) <= 7))
        arg_1_2 = ((array.index(value) >= 10) and (array.index(value) <= 11))
        
        if (arg_0_1 or arg_0_2):
            return 0
        if (arg_1_1 or arg_1_2):
            return 1
    except ValueError:
        np.nan

# Add column 'Insurance' and assign each row a binary Integer based on if the
# service of that appointment is an insurance massage
ds = appointmentData['a_Service']
appointmentData['Insurance'] = ds.apply(lambda x: find_insurance(x, services))

# %% Add column 'Status' and assign Integer based on appointments Status

ds = appointmentData['a_Status']
appointmentData.loc[:, 'Status'] = np.where(ds == 'PAID', 1,
                                       np.where(ds == 'DUE', 1,
                                       np.where(ds == 'CANCELLED', 2,
                                       np.where(ds == 'NO SHOW', 3,
                                       np.nan))))

# %% Add column 'No_Show' and assign Binary Integer based on if the client
#    No-Showed the appointment

ds = appointmentData['a_Status']
appointmentData.loc[:, 'No_Show'] = np.where(ds == 'NO SHOW', 1, 0)

# %% Add column 'Cancellation' and assign Binary Integer based on if the
#    client cancelled the appointment

ds = appointmentData['a_Status']
appointmentData.loc[:, 'Cancelled'] = np.where(ds == 'CANCELLED', 1, 0)

# %% Remove unused columns

del appointmentData['Date']
del appointmentData['Str_Day_of_Week']
del appointmentData['a_Service']
del appointmentData['a_Status']

print('----appointment data prepared')

# %% Join dataframes and remove columns that may reduce accuracy

# join clients table to appointments table by account number
joinedData = pd.merge(appointmentData, clientData, on='Account_Number', how='left')

print('-----client data joined to appointment data')

# remove columns that may reduce accuracy of prediction
del joinedData['Account_Number']
status = joinedData.pop('Status') # saving this for visualization
del joinedData['Year']
del joinedData['Num_Appts']

print('------data prepared')

# %% Copy joinedData to train to different models, one for No_Show and one
#    for Cancellations and remove their opposing columns

# DataFrame to predict No_Show
joinedData_n = joinedData.copy()
del joinedData_n['Cancelled']

# DataFrame to predict Cancellation
joinedData_c = joinedData.copy()
del joinedData_c['No_Show']

# %% Normalize dataframes to range between 0 and 1

# assign for normalizing
# these will be used later on user input
jdn = joinedData_n
jdc = joinedData_c

# normalize
joinedData_n = (jdn - jdn.min()) / (jdn.max() - jdn.min())
joinedData_c = (jdc - jdc.min()) / (jdc.max() - jdc.min())

print('-------data normalized')

# %% Set dependent variables for training

# No_Shows
x_n = joinedData_n.drop('No_Show', axis=1)
y_n = joinedData_n['No_Show']

# Cancellations
x_c = joinedData_c.drop('Cancelled', axis=1)
y_c = joinedData_c['Cancelled']

# %% Choose a random_state number to apply to training

rs = 42

# %% Split data sets into train and test samples
#    Note: Splitting 9:1 seemed to have the best AUC-ROC results

x_train_n, x_test_n, y_train_n, y_test_n = train_test_split(x_n, y_n, test_size=0.1, random_state=rs)
x_train_c, x_test_c, y_train_c, y_test_c = train_test_split(x_c, y_c, test_size=0.1, random_state=rs)

# convert to numpy arrays for cross validation
x_train_n = x_train_n.values # if hasattr(x_train_n, 'values') else x_train_n
y_train_n = y_train_n.values # if hasattr(y_train_n, 'values') else y_train_n


print('--------data split into training and test samples')

# %% Apply SMOTETomek to data to balance data

smt = SMOTETomek(random_state=rs)
x_train_n, y_train_n = smt.fit_resample(x_train_n, y_train_n)
x_train_c, y_train_c = smt.fit_resample(x_train_c, y_train_c)

print('---------training data resampled')

# %% Train and Calibrate RandomForestClassifier Algorithms

# base estimator
rfc_n = RandomForestClassifier(n_estimators=500, #min_weight_fraction_leaf=0.001,
                             random_state=rs, class_weight='balanced')
rfc_c = RandomForestClassifier(n_estimators=500, #min_weight_fraction_leaf=0.001,
                             random_state=rs, class_weight='balanced')

# training strategy
cccv_n = CalibratedClassifierCV(rfc_n, method='sigmoid')
cccv_c = CalibratedClassifierCV(rfc_c, method='sigmoid')

print(' ')
print('training and calibrating Random Forest (approximately: 1.5 minutes)')
print('...')

# train model
cccv_n.fit(x_train_n, y_train_n)
cccv_c.fit(x_train_c, y_train_c)

print('random forests trained and calibrated')
print(' ')

# %% Get models AUC-ROC scored

y_test_proba_n = cccv_n.predict_proba(x_test_n)[:, 1]
auc_roc_n = roc_auc_score(y_test_n, y_test_proba_n)
print('No-Show ' + f'AUC-ROC: {auc_roc_n}')

y_test_proba_c = cccv_c.predict_proba(x_test_c)[:, 1]
auc_roc_c = roc_auc_score(y_test_c, y_test_proba_c)
print('Cancellation ' + f'AUC-ROC: {auc_roc_c}')

# %% Get cross-validation scored
#    Note: using cv=2 to speed up method

# Cross-validation strategy
skf = StratifiedKFold(n_splits=2)
mean_fpr = np.linspace(0, 1, 100)

print(' ')
print('running cross-validation test on No-Show Model (approximately: 1 minute)')
print('...')

tprs = [] # Store true positive ratings
aucs = [] # Store AUCs

# For each split, fold data and measure auc
for train, test in skf.split(x_train_n, y_train_n):
    
    # assign folded data
    x_train_fold = x_train_n[train]
    y_train_fold = y_train_n[train]
    x_test_fold = x_train_n[test]
    y_test_fold = y_train_n[test]

    # Train and predict current fold
    cccv_n.fit(x_train_fold, y_train_fold)
    y_pred_proba_skf = cccv_n.predict_proba(x_test_fold)[:, 1]
    
    # calculate roc_curve for current fold and store true positive ratings
    fpr_skf, tpr_skf, _ = roc_curve(y_test_fold, y_pred_proba_skf)
    tprs.append(np.interp(mean_fpr, fpr_skf, tpr_skf))
    tprs[-1][0] = 0.0
    
    #calculate area under the ROC curve and store it
    roc_auc_skf = auc(fpr_skf, tpr_skf)
    aucs.append(roc_auc_skf)

# Calculate mean and standard deviation of the AUCs
mean_tpr_n = np.mean(tprs, axis=0)
mean_tpr_n[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr_n)
std_auc_n = np.std(aucs)

print(' ')
print(f'Cross-validated No-Show AUC-ROC scores: {aucs}')
print(f'Mean No-Show AUC-ROC: {mean_auc}')

print(' ')
print('running cross-validation test on Cancellation Model (approximately: 1 minute)')
print('...')

tprs = [] # Store true positive ratings
aucs = [] # Store AUCs

# For each split, fold data and measure auc
for train, test in skf.split(x_train_n, y_train_n):
    
    # assign folded data
    x_train_fold = x_train_n[train]
    y_train_fold = y_train_n[train]
    x_test_fold = x_train_n[test]
    y_test_fold = y_train_n[test]

    # Train and predict current fold
    cccv_n.fit(x_train_fold, y_train_fold)
    y_pred_proba_skf = cccv_n.predict_proba(x_test_fold)[:, 1]
    
    # calculate roc_curve for current fold and store true positive ratings
    fpr_skf, tpr_skf, _ = roc_curve(y_test_fold, y_pred_proba_skf)
    tprs.append(np.interp(mean_fpr, fpr_skf, tpr_skf))
    tprs[-1][0] = 0.0
    
    #calculate area under the ROC curve and store it
    roc_auc_skf = auc(fpr_skf, tpr_skf)
    aucs.append(roc_auc_skf)

# Calculate mean and standard deviation of the AUCs
mean_tpr_c = np.mean(tprs, axis=0)
mean_tpr_c[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr_c)
std_auc_c = np.std(aucs)

print(' ')
print(f'Cross-validated Cancellation AUC-ROC scores: {aucs}')
print(f'Mean Cancellation AUC-ROC: {mean_auc}')

# %% Choose if we're predicting No_Show or Cancellation

# Options:
#   1. 'No_Show'
#   2. 'Cancelled'
to_predict = 'Cancelled'

# %% Get inputs

#input for prediction
user_input = {
    'Practitioner' : [rand.randint(1,3)],
    'Quarter' : [rand.randint(1,4)],
    'Month' : [rand.randint(1,12)],
    'Day_of_Week' : [rand.randint(1,31)],
    'Hour' : [rand.randint(9,19)],
    'Service' : [rand.randint(1,11)],
    'Length' : [rand.randint(1,3) * 30],
    'Insurance' : [rand.randint(0,1)],
    to_predict : [0],
    'Zip' : [int(clientData['Zip'].mode()[0])],
    'Sex' : [rand.randint(1,2)],
    'Age' : [rand.randint(18,90)],
    'Marital' : [rand.randint(1,6)],
    'Employment' : [rand.randint(1,5)],
    'Accident' : [rand.randint(0,1)],
    }

# %% Display Visualizations based on what prediction we're making

if to_predict == 'No_Show':
    jd = jdn
    cccv = cccv_n
    y_test_proba = y_test_proba_n
    y_test = y_test_n
    mean_tpr = mean_tpr_n
    std_auc = std_auc_n
    
if to_predict == 'Cancelled':
    jd = jdc
    cccv = cccv_c
    y_test_proba = y_test_proba_c
    y_test = y_test_c
    mean_tpr = mean_tpr_c
    std_auc = std_auc_c
    
# %% Make prediction using user_input

# turn input into dataframe
input_frame = pd.DataFrame(user_input)

# normalize input
input_frame = (input_frame - jd.min()) / (jd.max() - jd.min())

# remove dependency
del input_frame[to_predict]

# Make prediction
predictions = cccv.predict_proba(input_frame)
print(' ')
print(f'Chance of {to_predict}: {predictions[0, 1] * 100:.2f}%')

# %% Feature Importance Visualization

# Get Feature importances from estimator (Random Forest)
importances = cccv.calibrated_classifiers_[0].estimator.feature_importances_
indices = np.argsort(importances)

# Create plot
plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [x_n.columns[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# %% Distribution of probabilities visualization

# Show distribution of probabilities
sns.histplot(y_test_proba, kde=True)
plt.xlabel('Predicted Probability of ' + to_predict)
plt.title('Distribution of Predicted Probabilities')
plt.show()

# %% ROC_Curve

# Create Roc_curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)

# Show ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# plot the mean cross-validation roc curve
plt.plot(mean_fpr, mean_tpr, color='blue', lw=2, linestyle='-', label=f'Mean CV ROC curve (area = {mean_auc:.2f} ± {std_auc:.2f})')

# build plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# %% Status visualization

# Get counts of each status
status_counts = appointmentData['Status'].value_counts()

# Create custom labels
custom_labels = ['Completed Appointments', 'Cancelled Appointments', 'No-Shows']

# Set up plot
plt.figure(figsize=(8,8))
status_counts.plot.pie(labels=custom_labels, autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'skyblue', 'lightcoral'])
plt.title('Appointment Status Distribution')
plt.ylabel('')  # Hide the y-label
plt.show()

# %% Zip Codes visualization

# Get zip code counts
zip_counts = clientData['Zip'].astype(int).value_counts()
top_15 = zip_counts.nlargest(15)
Other = zip_counts.iloc[15:].sum()
top_15['Other'] = Other

# Set up plot
plt.figure(figsize=(10, 6))
top_15.plot(kind='bar', color='skyblue')
plt.xlabel('Zip Code')
plt.ylabel('Frequency')
plt.title('Zip Codes with Highest Value Counts')
plt.xticks(rotation=45)
plt.show()

# %% Age Buckets visualization

# Put Status back in Joined Data
joinedData.insert(0, 'Status', status)
df = joinedData

# Create age buckets
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
labels = ['1-10', '11-20', '21-30', '31-40', '41-50',
          '51-60', '61-70', '71-80', '81-90', '91-100']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Count the occurrences of each status within each age bucket
age_status_counts = df.groupby(['Age_Group', 'Status'], observed=False).size().reset_index(name='Count')

# Plot the bar chart
plt.figure(figsize=(12, 8))
sns.barplot(data=age_status_counts, x='Age_Group', y='Count', hue='Status', palette='viridis')
plt.xlabel('Age Group')
plt.ylabel('Number of Appointments')
plt.title('Appointment Status by Age Group')
plt.xticks(rotation=45)

# Change Key names in legend
new_labels = ['Completed Appointments', 'Cancelled Appointments', 'No-Shows']
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, new_labels, title='Appointment Status')

plt.show()
