from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load and prepare data
data = pd.read_csv('/path/to/your/data.csv')  # Update this path

# Clean the data by converting Debit and Credit to numeric, handling errors
data['Debit'] = pd.to_numeric(data['Debit'], errors='coerce').fillna(0)
data['Credit'] = pd.to_numeric(data['Credit'], errors='coerce').fillna(0)

# Calculate total income and total expenses
total_income = data['Credit'].sum()
total_expenses = data['Debit'].sum()

# Prepare the dataset for training
savings_goal = 1000  # Set your savings goal here

# Create a DataFrame for the features and labels
features = pd.DataFrame({
    'income': [total_income],
    'expenses': [total_expenses]
})
labels = pd.Series([savings_goal])

# Split data and train the model
# Here, you can only train a model with more than one example; otherwise, it won’t generalize
if len(features) > 1:
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
else:
    model = LinearRegression()  # Create model but not fit it as you need more data

@app.route('/api/savings', methods=['GET'])
def get_savings():
    savings_data = {
        "savingsGoal": savings_goal,
        "currentSavings": 300,  # Replace with actual value if available
        "savingsProgress": (300 / savings_goal) * 100  # Example calculation
    }
    return jsonify(savings_data)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    income = data.get('income')
    expenses = data.get('expenses')

    if income is None or expenses is None:
        return jsonify({'error': 'Income and expenses are required'}), 400

    prediction = model.predict([[income, expenses]]) if len(features) > 1 else savings_goal  # Fallback if model not trained
    return jsonify({'savings_goal_prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
