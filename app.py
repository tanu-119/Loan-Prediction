from flask import Flask, request, render_template
import pickle
import numpy as np 
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
        Gender = request.form.get('Gender')
        Married = request.form.get('Married')
        Education = request.form.get('Education')
        Self_Employed = request.form.get('Self_Employed')  
        ApplicantIncome = int(request.form.get('ApplicantIncome'))  
        CoapplicantIncome = int(request.form.get('CoapplicantIncome')) 
        LoanAmount = float(request.form.get('LoanAmount'))   
        Loan_Amount_Term = int(request.form.get('Loan_Amount_Term'))   
        Credit_History = float(request.form.get('Credit_History'))   
        Property_Area = request.form.get('Property_Area')  

        # Create input array
        test_data = [[Gender, Married, Education, Self_Employed, ApplicantIncome,
                      CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area]]  
        ord_enc = OrdinalEncoder()
        test_data[["Gender",'Married','Education','Self_Employed','Property_Area','Loan_Status']] = ord_enc.fit_transform(test_data[["Gender",'Married','Education','Self_Employed','Property_Area','Loan_Status']])
        test_data[["Gender",'Married','Education','Self_Employed','Property_Area','Loan_Status','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']] = test_data[["Gender",'Married','Education','Self_Employed','Property_Area','Loan_Status','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']].astype('int')

        # Predict
        prediction = model.predict(test_data)

        # Format result
        result = "Approved" if prediction[0] == 1 else "Denied"
        return render_template('form.html', result=result)
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__': 
    app.run(debug=True) 
