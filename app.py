from flask import Flask, request, render_template
import pickle
import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__) 
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/') 
def home(): 
    return render_template('index.html') 
@app.route('/form')
def form():
    return render_template('form.html') 
@app.route('/predict/', methods=['GET', 'POST'])
def predict():  
    if request.method == 'POST': 
        Gender = request.form.get('Gender')
        Married = request.form.get('Married')
        Education = request.form.get('Education')
        Self_Employed = request.form.get('Self_Employed')  
        ApplicantIncome = request.form.get('ApplicantIncome')  
        CoapplicantIncome = request.form.get('CoapplicantIncome') 
        LoanAmount = request.form.get('LoanAmount')   
        Loan_Amount_Term = request.form.get('Loan_Amount_Term')   
        Credit_History = request.form.get('Credit_History')   
        Property_Area = request.form.get('Property_Area')  
        test_data = [[Gender, Married, Education, Self_Employed, ApplicantIncome,CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History,Property_Area] ]  
        result = model.predict(test_data) 

    return render_template('form.html', result=result) 

if __name__ == '__main__': 
    app.run(debug=True) 
