import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to clean data
def cleanData(dataset):
    for column in dataset.columns:
        if dataset[column].dtype == type(object):
            le = LabelEncoder()
            dataset[column] = le.fit_transform(dataset[column])
    return dataset

# Function to split features and labels
def split_feature_class(dataset, feature):
    features = dataset.drop(feature, axis=1)
    labels = dataset[feature].copy()
    return features, labels

# Load data
dataset = pd.read_csv("loan.csv")

# Clean data
dataset = cleanData(dataset)

# Split features and labels
features, labels = split_feature_class(dataset, "Loan_Status")

# Split train 80%, test 20%
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state=0
)

# Impute missing values
imputer = SimpleImputer(strategy="mean")
train_features_imputed = imputer.fit_transform(train_features)
test_features_imputed = imputer.transform(test_features)

# Model
model = GaussianNB()
#train
model.fit(train_features_imputed, train_labels)

#predict
clf_pred = model.predict(test_features_imputed)
#Accuracy Score
print("Accuracy = ", accuracy_score(test_labels, clf_pred))

#----------------   predictive system --------------------

# Function to predict loan status based on user input
def predict_loan_status(applicant_data):
    # Convert user input to DataFrame
    input_data = pd.DataFrame([applicant_data], columns=features.columns)
    
    # Clean and impute user input
    input_data = cleanData(input_data)
    input_data_imputed = imputer.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_imputed)
    
    return "Approved" if prediction[0] == 1 else "Not Approved"

# User input
user_input = {
    'Gender': input("Enter Gender (Male or Female): "),
    'Married': input("Enter Marital Status (Yes or No): "),
    'Dependents': int(input("Enter Dependents (Number): ")),
    'Education': input("Enter Education (Graduate or Not Graduate): "),
    'Self_Employed': input("Enter Self Employed (Yes or No): "),
    'ApplicantIncome': float(input("Enter Applicant's Income: ")),
    'CoapplicantIncome': float(input("Enter Coapplicant's Income: ")),
    'LoanAmount': float(input("Enter Loan Amount: ")),
    'Loan_Amount_Term': float(input("Enter Loan Amount Term (in months): ")),
    'Credit_History': float(input("Enter Credit History (1 for good, 0 for bad): ")),
    'Property_Area': input("Enter Property Area (e.g., Urban, Rural, Semiurban): "),
}

# Predict loan status
result = predict_loan_status(user_input)
print("Loan Status:", result)


