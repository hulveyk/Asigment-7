import pandas as pd 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LogisticRegression 

  

# Load the dataset from the spreadsheet 

file_path = r'C:\Users\Hulveyk03\Downloads\baseball.xlsx' 

data = pd.read_excel(file_path) 

  

# Display the first few rows of the dataset to understand its structure 

print("First few rows of the dataset:") 

print(data.head()) 

  

# Extracting features (X) and target variable (y) 

features = ['Runs Scored', 'Runs Allowed', 'Wins', 'OBP', 'SLG', 'Team Batting Average'] 

X = data[features] 

y = data['Playoffs'] 

  

# Split the data into training and testing sets 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

  

# Create a logistic regression model 

model = LogisticRegression() 

  

# Train the model on the training data 

model.fit(X_train, y_train) 

  

# Print out the coefficients of the model for each feature 

print("Model Coefficients:") 

for feature, coef in zip(features, model.coef_[0]): 

    print(f"{feature}: {coef}") 

  

# Print out the prediction model 

print("Prediction Model:") 

print(model) 

  

# Evaluate the model on the testing data 

accuracy = model.score(X_test, y_test) 

print("Model Accuracy:", accuracy) 
print("Go Brewers!")
