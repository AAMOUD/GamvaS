import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class SavingsModel:
    def __init__(self, data_file):
        self.data_file = data_file
        self.model, self.savings_goal = self.train_model()
        self.current_savings = 300  # Set the current savings amount (replace with actual if available)

    def train_model(self):
        # Load data
        data = pd.read_csv(self.data_file)  # Ensure this path is correct
        print("Data Loaded Successfully:")
        print(data.head())  # Print the first few rows of the DataFrame

        # Clean the data by converting Debit and Credit to numeric, handling errors
        data['Debit'] = pd.to_numeric(data['Debit'], errors='coerce').fillna(0)
        data['Credit'] = pd.to_numeric(data['Credit'], errors='coerce').fillna(0)

        # Print the cleaned data
        print("Cleaned Data:")
        print(data[['Debit', 'Credit']].head())  # Show only Debit and Credit columns

        # Prepare the dataset for training
        savings_goal = 1000  # Set your savings goal here

        # Create a DataFrame for features based on rows of data
        features = data[['Credit', 'Debit']].copy()
        features['savings_goal'] = savings_goal  # Add savings goal as a constant feature

        # Prepare labels (e.g., the target savings for each transaction)
        labels = pd.Series([savings_goal] * len(features))  # Create a label for each row

        print("Features and Labels Prepared:")
        print(features.head())  # Print features only

        # Split data and train the model
        X_train, X_test, y_train, y_test = train_test_split(features[['Credit', 'Debit']], labels, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate Mean Squared Error
        mse = mean_squared_error(y_test, predictions)
        print("Mean Squared Error:", mse)  # Print MSE

        return model, savings_goal  # Return the trained model and savings goal

    def predict(self, income, expenses):
        predicted_savings = income - expenses  # Calculate predicted savings

        # If the predicted savings are positive
        if predicted_savings > 0:
            # Estimate how long it will take to reach the savings goal
            time_to_goal = (self.savings_goal - self.current_savings) / predicted_savings
            return {
                "predicted_savings": predicted_savings,
                "time_to_goal_months": max(0, time_to_goal)  # Avoid negative time
            }
        else:
            return {
                "predicted_savings": predicted_savings,
                "time_to_goal_months": float('inf')  # Indicates overspending
            }

if __name__ == '__main__':
    model = SavingsModel('data.csv')  # Change this if needed
