from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model and scaler
model = pickle.load(open('loan_risk_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Load the pre-trained scaler
le_home = LabelEncoder()
le_intent = LabelEncoder()

# Column names to match the model's input format
feature_names = ['Age', 'Income', 'Home', 'Emp_length', 'Intent', 'Amount', 'Cred_length', 'Rate']

@app.route('/')
def home():
    return render_template('index.html')  # Display the index page initially

@app.route('/form')
def form():
    return render_template('form.html')  # Display the form page when the user clicks on "Get Loan Prediction"

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    age = int(request.form['age'])
    income = float(request.form['income'])
    home = request.form['home']
    emp_length = int(request.form['emp_length'])
    intent = request.form['intent']
    amount = float(request.form['amount'])
    rate = float(request.form['rate'])
    cred_length = int(request.form['cred_length'])

    # Encode 'Home' and 'Intent' using label encoders
    home_encoded = le_home.fit_transform([home])[0]
    intent_encoded = le_intent.fit_transform([intent])[0]

    # Create a DataFrame with the user's input
    input_data = pd.DataFrame([{
        'Age': age,
        'Income': income,
        'Home': home_encoded,
        'Emp_length': emp_length,
        'Intent': intent_encoded,
        'Amount': amount,
        'Cred_length': cred_length,
        'Rate': rate
    }], columns=feature_names)

    input_data[['Income', 'Amount', 'Rate']] = scaler.transform(input_data[['Income', 'Amount', 'Rate']])


    # Predict the risk of the loan
    prediction = model.predict(input_data)
    result = "Approved" if prediction[0] == 1 else "Denied"

    # Return the result to the form page
    return render_template('form.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
