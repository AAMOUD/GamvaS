from flask import Flask, request, jsonify
from ml_model import SavingsModel  # Ensure your ml_model.py contains a class named SavingsModel

app = Flask(__name__)
model = SavingsModel('data.csv')  # Ensure the path is correct

@app.route('/api/savings', methods=['GET'])
def get_savings():
    savings_data = {
        "savingsGoal": model.savings_goal,
        "currentSavings": model.current_savings,  # Use actual current savings
        "savingsProgress": (model.current_savings / model.savings_goal) * 100  # Example calculation
    }
    return jsonify(savings_data)

@app.route('/api/predict', methods=['POST'])  # Consistent endpoint
def predict():
    data = request.get_json()
    income = data.get('income')
    expenses = data.get('expenses')

    if income is None or expenses is None:
        return jsonify({'error': 'Income and expenses are required'}), 400

    # Predict based on user input
    prediction = model.predict(income, expenses)
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)
