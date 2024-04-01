import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load your CSV file
data = pd.read_csv('C:\\Users\\aicha\\Documents\\GamvaS\\flask-backend\\data.csv')  # Update this path

# Preprocess the data
data = data.dropna()  # Drop rows with missing values
features = data[['income', 'expenses']]
labels = data['savings_goal']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)
