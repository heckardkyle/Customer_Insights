# CustomerInsights

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/heckardkyle/CustomerInsights/HEAD) <br>
https://mybinder.org/v2/gh/heckardkyle/CustomerInsights/HEAD

The repository contains a jupyter notebook name Customer Insights.
I made this repository as part of my Computer Science Capstone project.

The purpose of the notebook is to take CSVs, train a machine learning model and predict the chance of a client No-Showing or Cancelling an appointments.

By clicking the badge or link above, it will create a temporary URL in <b>mybinder.org</b> from my repository.

### Clients.csv
This is a dataset that contains client information such as account_number, gender, age, zip code, employment status, etc., used to build the machine learning model.

### Appointments.csv
This is a dataset that contain a list of appointments containing information such as the date and time of appointment, status, the account_number associated, etc.

### CustomerInsights.py
#### Environment
This is the original file used to prepare the data and train the machine learning model.
I developoed this in the <b>Spyder 6 IDE</b> using <b>Python 3.12.3</b>.
#### Goal
The goal of the program is to take Client and Appointment characteristics and train a model that can predict the percent chance of a client No-Showing or Cancelling an appointment.
My <b>hypothesis</b> is that <b>age</b> is the main driving factor when determining if they will No-Show or Cancel
#### Data Prep
I prepared the <i>Clients.csv</i> and <i>Appointments.csv</i> datasets by converting all characteristics to integers, filling nulls with random values based on value frequencies, removing unnecessary columns, extracting information from things like the timestamp to make more columns, then joining the clients dataset to the appointments dataset via Account Number.<br>
Afterwords I normalized the data so that all columns range from 0 to 1 to improve Machine learning model performance and used <b>SMOTETomek</b> to resample and balance the data.
#### ML Method
This is a small dataset, so the method I chose to use is the <b>Random Forest Classifier</b> method to get the best performance and prevent overfitting.
I used <b>CalibratedClassifierCV()</b> to calibrate the model and a </b>StratifiedKFold</b> using 2 k-folds (for performance reason. The default 5 k-folds took too long) to cross-validate the model.
Before cross validation, the model had an <b>AUC-ROC</b> score of about 70%. After, it was about 99%.
#### Visualization
The program includes visualiztions to help provide insights on client behavior.
This includes distribution of the zip codes most clients reside, appointment status distribution, appointment status by age group, and feature importance.
I also included a distribution of probabilites and a plot with the ROC curve to show the model performance.
#### Knowledge gained
We did learn that the retirement age group was the most likely to make it to their appointments, but the distribution was pretty proportionate across all of the age groups.
The feature importance visualization showed us that the clients' <b>zip code</b> of residence was the main factor in determining if the client no-shows or cancels an appointment.
This new knowledge disproved my hypothesis that age was the main factor.

### CustomerInsights.ipynb
This is a <b>jupyter notebook</b> file designed to run in <i>binder.org</i> that has an "application" built into the first code cell that can be used to enter user information and make a prediction on the percent chance of that person no-showing or cancelling.
The application works by calling <i>CIApplication.py</i> from the first cell.
The notebook also includes a walkthrough of all of my code from CustomerInsights.py showing how I developed my model.
Running each cell shows the code's output.

### CIApplication.py
This is the program the first cell of CustomerInsights.ipynb calls to start the "application".
Because binder.org always opens the notebook in a fresh environment, the necessary packages need to be reintstalled each time.
So, the first thing the application does is check if packages are installed or not and installs them.
Then, it imports and preps the data, and trains the model (Most of this is done the exact same as in <i>CustomerInsights.py</i>).
Afterwords, a UserForm is displayped in the Cell output for the user to enter the information and Make a prediction.
The cell will then output the prediction and displaying the same visualizations as in <i>CustomerInsights.py</i>.
The userform and visualizations are displayed using <b>ipywidgets</b>.
