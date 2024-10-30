#CIApplication.py

import importlib.util
import subprocess
import sys
from IPython.display import clear_output
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

#################################################
#################################################
############# Install Packages ##################
#################################################
#################################################

def install_and_import(package):
    if importlib.util.find_spec(package) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    else:
        print(f"{package} is already installed")

print('Installing necessary packages')
print('(Approximately _ minutes)')
print(' ')

install_and_import('pandas')
install_and_import('matplotlib')
install_and_import('seaborn')
install_and_import('scikit-learn')
install_and_import('imblearn')

clear_output()
print('Finished installing packages')
print(' ')
print('Training Data')

############################################
############################################
########### Train Model ####################
############################################
############################################

# %% Load Raw Data (csv)

print('Importing Data and training model')
print('(Approximately 4 minutes)')

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
clientData = clientData_filtered.copy()

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
x_train_n = x_train_n.values
y_train_n = y_train_n.values


print('--------data split into training and test samples')

# %% Apply SMOTETomek to data to balance data

smt = SMOTETomek(random_state=rs)
x_train_n, y_train_n = smt.fit_resample(x_train_n, y_train_n)
x_train_c, y_train_c = smt.fit_resample(x_train_c, y_train_c)

print('---------training data resampled')

# %% Train and Calibrate RandomForestClassifier Algorithms

# base estimator
rfc_n = RandomForestClassifier(n_estimators=500, random_state=rs, 
                               class_weight='balanced')
rfc_c = RandomForestClassifier(n_estimators=500, random_state=rs, 
                               class_weight='balanced')

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

# %% Cross-validate model
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
mean_auc_n = auc(mean_fpr, mean_tpr_n)
std_auc_n = np.std(aucs)

print(' ')
print(f'Cross-validated No-Show AUC-ROC scores: {aucs}')
print(f'Mean No-Show AUC-ROC: {mean_auc_n}')

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
mean_auc_c = auc(mean_fpr, mean_tpr_c)
std_auc_c = np.std(aucs)

print(' ')
print(f'Cross-validated Cancellation AUC-ROC scores: {aucs}')
print(f'Mean Cancellation AUC-ROC: {mean_auc_c}')

clear_output()
print('Machine-learning Model Trained')

# UserInput.py

import ipywidgets as widgets
from IPython.display import display, clear_output
from datetime import datetime
import pandas as pd

# find length of service
def in_find_length(value):
    try:
        arg_30 = (value == 0)
        arg_60 = ((value >= 1) and (value <= 7))
        arg_90 = ((value >= 8) and (value <= 11))
        
        if arg_30:
            return 30
        if arg_60:
            return 60
        if arg_90:
            return 90
    except ValueError:
        np.nan

# find if service is insurance
def in_find_insurance(value):
    try:
        arg_0_1 = ((value >= 0) and (value <= 3))
        arg_0_2 = ((value >= 8) and (value <= 9))
        arg_1_1 = ((value >= 4) and (value <= 7))
        arg_1_2 = ((value >= 10) and (value <= 11))
        
        if (arg_0_1 or arg_0_2):
            return 0
        if (arg_1_1 or arg_1_2):
            return 1
    except ValueError:
        np.nan

# when ignore_zip is checked, disable zip_input
def update_zip_input(change):
    zip_input.disabled = change['new']

# Function to make predictions
def make_prediction(button):
    # get non-date inputs
    in_practitioner = practitioner_input.value
    in_date = date_input.value
    in_hour = time_input.value
    in_service = service_input.value
    in_zip = zip_input.value
    in_sex = sex_input.value
    in_age = age_input.value
    in_marital = marital_input.value
    in_employment = employment_input.value
    in_accident = accident_input.value

    # If not selected, fill with modes
    if in_practitioner == 0:
        print('pract')
    if in_service == -1:
        in_service = 1
        print('serv')
    if ignore_zip.value == True:
        print('zip')
    if in_sex == 0:
        print('sex')
    if in_age == 0:
        print('age')
    if in_marital == 0:
        print('marital')
    if in_employment == 0:
        print('employment')
    if in_accident == -1:
        print('accident')

    if in_date is None:
        print('date')
        in_quarter = 0
        in_month = 0
        in_day_of_week = 0
    else:
        in_date = pd.to_datetime(in_date, format='%m/%d/%Y')
    
        # extract quarter, month and dayofweek
        in_quarter = in_date.quarter
        in_month = in_date.month
        in_day_of_week = in_date.day_name()
    
        # convert dayofweektointeger
        if in_day_of_week == 'Sunday':
            in_day_of_week = 0
        if in_day_of_week == 'Monday':
            in_day_of_week = 1
        if in_day_of_week == 'Tuesday':
            in_day_of_week = 2
        if in_day_of_week == 'Wednesday':
            in_day_of_week = 3
        if in_day_of_week == 'Thursday':
            in_day_of_week = 4
        if in_day_of_week == 'Friday':
            in_day_of_week = 5
        if in_day_of_week == 'Saturday':
            in_day_of_week = 6

    in_length = in_find_length(in_service)
    in_insurance = in_find_insurance(in_service)

    no_show_output.value = 'yay'
    cancellation_output.value = 'yay'

    tab_widget.layout.display = 'block'  # Hide the tab widget initially

'''
# Example prediction function
def make_prediction(input_value):
    # Replace this with your actual prediction logic
    return f"Predicted value for {input_value}: {input_value * 2}"

# Function to handle button click
def on_button_click(b):
    input_value = input_widget.value
    prediction = make_prediction(input_value)
    output_widget.value = prediction  # Update the output widget with the prediction
    tab_widget.layout.display = 'block'  # Show the tab widget

    # Update tab content with prediction information
    tab_contents[0].children = [widgets.Label(value=prediction)]
    tab_contents[1].children = [widgets.Label(value=f"Additional info for {input_value}...")]

# Create input widget
input_widget = widgets.FloatText(
    description='Input:',
    value=0.0
)

# Create button widget
predict_button = widgets.Button(
    description='Make Prediction'
)

# Create output widget to display prediction
output_widget = widgets.Label(value='')

# Create tab widget
tab_contents = [widgets.VBox([]), widgets.VBox([])]  # Two tabs for different information
tab_widget = widgets.Tab(children=tab_contents)
tab_widget.set_title(0, 'Prediction Info')
tab_widget.set_title(1, 'Additional Info')
tab_widget.layout.display = 'none'  # Hide the tab widget initially

# Link button click event to the function
predict_button.on_click(on_button_click)

# Display the widgets
display(input_widget, predict_button, output_widget, tab_widget)
'''

# Choice box inputs
practitioners = [('-', 0),
                 ('John (1)', 1),
                 ('Jane (2)', 2),
                 ('Jean (3)', 3)
                ]

times = [('-', 0),
         ('8:00AM',8),
         ('9:00AM',9),
         ('10:00AM',10),
         ('11:00AM',11),
         ('12:00AM',12),
         ('1:00PM',13),
         ('2:00PM',14),
         ('3:00PM',15),
         ('4:00PM',16),
         ('5:00PM',17),
         ('6:00PM',18),
         ('7:00PM',19),
         ('8:00PM',20)
        ]

i_services = [('-', -1),
              ('30 minute 30 Minute Add-On',0),
              ('60 minute Deep Tissue Massage',1),
              ('60 minute Relaxation Massage',2),
              ('60 minute Pregnancy Massage',3),
              ('60 minute Insurance Massage',4),
              ('60 minute Pregnancy Insurance Massage',5),
              ('60 minute L&amp;I Massage',6),
              ('60 minute Motor Vehicle Accident (MVA) Massage',7),
              ('90 minute Deep Tissue Massage',8),
              ('90 minute Relaxation Massage',9),
              ('90 minute Insurance Massage',10),
              ('90 minute Pregnancy Insurance Massage',11)
             ]

genders = [('-',0),
           ('Male',1),
           ('Female',2),
           ('Other',3)
          ]

ages = [('-', -1)]  # Initialize the list with the first cell
for age in range(10, 101):
    birth_year = 2024 - age  # Calculate the birth year
    ages.append((f"{birth_year} ({age})", age))

maritals = [('-', 0),
            ('Married', 1),
            ('Common-Law', 2),
            ('Divorced', 3),
            ('Widowed', 4),
            ('Single', 5),
            ('Other', 6)
           ]

employments = [('-', 0),
               ('Employed', 1),
               ('Student', 2),
               ('Retired', 3),
               ('Unemployed', 4),
               ('Other', 5)
              ]

accidents = [('-', -1),
             ('No', 0),
             ('Yes', 1)
            ]  

# styles
label_style = {'font_weight': 'bold', 'font_size': '20px'}
output_style = {'font_weight': 'bold', 'font_size': '16px'}
long_desc = {'description_width': 'initial'}

# label widgets
appointment_label = widgets.Label(value='Appointment Info', style=label_style)
client_label = widgets.Label(value='Client Info', style=label_style)
empty_label1 = widgets.Label(value=' ')
empty_label2 = widgets.Label(value=' ')
empty_label3 = widgets.Label(value=' ')
empty_label4 = widgets.Label(value=' ')
empty_label5 = widgets.Label(value=' ')
empty_label6 = widgets.Label(value=' ')
no_show_output = widgets.Label(value=' ', style=output_style)
cancellation_output = widgets.Label(value=' ', style=output_style)

# Input widgets
practitioner_input = widgets.Dropdown(options=practitioners, description='Practitioner: ')
date_input = widgets.DatePicker(description='Date: (Open Mon-Sat)', style=long_desc)
time_input = widgets.Dropdown(options=times, description='Time: ')
service_input = widgets.Dropdown(options=i_services, description='Service: ')
sex_input = widgets.Dropdown(options=genders, description='Gender: ')
age_input = widgets.Dropdown(options=ages, description='Birth Year (Age): ', style=long_desc)
zip_input = widgets.BoundedIntText(value=98466, min=10000, max=99999, step=1, description='5-Digit Zip Code: ', disabled=False)
marital_input = widgets.Dropdown(options=maritals, description='Marital Status: ', style=long_desc)
employment_input = widgets.Dropdown(options=employments, description='Employment Status: ', style=long_desc)
accident_input = widgets.Dropdown(options=accidents, description='Motor Vehicle Accident (MVA): ', style=long_desc)

# Check box widget and link to zip_input
ignore_zip = widgets.Checkbox(value=False, description='Ignore Zip? ', disabled=False, indent=False)
ignore_zip.observe(update_zip_input, names='value')

# Button Widget
predict_button = widgets.Button(description='Make Prediction')

# widget columns
column1 = widgets.VBox([appointment_label,
                        empty_label1,
                        practitioner_input,
                        date_input,
                        time_input,
                        service_input
                        ])

column2 = widgets.VBox([client_label,
                        empty_label2,
                        sex_input,
                        age_input,
                        zip_input,
                        marital_input,
                        employment_input,
                        accident_input
                       ])

column3 = widgets.VBox([empty_label3,
                       empty_label4,
                       empty_label5,
                       empty_label6,
                       ignore_zip
                      ])

# Create a divider with a fixed width
vertical_divider = widgets.VBox(layout=widgets.Layout(width='60px'))
horizontal_divider1 = widgets.HBox(layout=widgets.Layout(height='30px'))
horizontal_divider2 = widgets.HBox(layout=widgets.Layout(height='30px'))

# User input layout
column_layout = widgets.HBox([column1, vertical_divider, column2, column3])
inputs_layout = widgets.VBox([column_layout,
                              predict_button,
                              horizontal_divider1,
                              no_show_output,
                              cancellation_output,
                              horizontal_divider2
                             ])

tab_contents = [widgets.VBox([]), widgets.VBox([])]  # Two tabs for different information
tab_widget = widgets.Tab(children=tab_contents)
tab_widget.set_title(0, 'Prediction Info')
tab_widget.set_title(1, 'Additional Info')
tab_widget.layout.display = 'none'  # Hide the tab widget initially

# Display Layouts
display(inputs_layout,tab_widget)

predict_button.on_click(make_prediction)