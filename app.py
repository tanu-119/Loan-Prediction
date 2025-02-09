from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

app = Flask(__name__) 
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/') 
def home(): 
    return render_template('index.html') 

@app.route('/form')
def form():
    return render_template('form.html') 

@app.route('/predict/', methods=['POST'])
def predict():
    try:
        # Collect input values and convert them to integers
        data = {
            'Gender': request.form.get('Gender'),
            'Married': request.form.get('Married'),
            'Education': request.form.get('Education'),
            'Self_Employed': request.form.get('Self_Employed'),
            'ApplicantIncome': int(request.form.get('ApplicantIncome')),
            'CoapplicantIncome': int(request.form.get('CoapplicantIncome')),
            'LoanAmount': int(float(request.form.get('LoanAmount'))),  # Converted from float to int
            'Loan_Amount_Term': int(request.form.get('Loan_Amount_Term')),
            'Credit_History': int(float(request.form.get('Credit_History'))),  # Converted from float to int
            'Property_Area': request.form.get('Property_Area')
        }

        # Convert dictionary to DataFrame
        df = pd.DataFrame([data])

        # Encoding categorical variables
        categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
        encoder = OrdinalEncoder()
        df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

        # Convert everything to integer
        df = df.astype('int64')

        # Predict using the model
        prediction = model.predict(df)

        # Format result
        result = "Approved" if prediction[0] == 1 else "Denied"
        return render_template('form.html', result=result)
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__': 
    app.run(debug=True)
