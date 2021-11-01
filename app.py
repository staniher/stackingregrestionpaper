from flask import Flask,request,render_template
import numpy as np
app=Flask(__name__)

@app.route('/')
#The below function returns the index.html page of our project
def home():
    return render_template('index.html')
# "predict" will be specified in the html form as an action to allow the predict method to be called 
@app.route('/predict', methods=['POST']) 
#The function below makes the prediction
def predict():
    import joblib
    #We load our saved model in the Flask project to use it
    model=joblib.load('stackedReg.ml')
    #We get all the values entered in the html form as a list
    string_features=[i for i in request.form.values()]
    #string_features=["0","12","54","0",'11/30/2021']
    #We get the last value of the list (i.e. date of entry in the hospital)
    date_hospitalisation =string_features[-1]  
    #string_features[0].
    #We get all the values of the list (Gender, Age, Disease, Service) except the last one 
    features_model = [string_features[0],string_features[1],string_features[2],string_features[3]]
    #We convert all our features to float
    features_model=[float(j) for j in features_model]
    #We reshape the features to make it an np vector able to be introduced in the model for prediction
    features_model=np.array([features_model]).reshape(1,4)
    #We predict using the model that has been loaded above
    prediction=model.predict(features_model)[0]
    #We convert the date of entry in the hospital in datetime to help us to draw the day, the month, the year for an easier exit of prediction
    import pandas as pd
    date_entry_hopital=pd.to_datetime(date_hospitalisation)
    from datetime import datetime,timedelta
    #We add the predicted days of hospital discharge to the entered date of hospital entry to find the discharge date
    date_discharge_hopital = date_entry_hopital + timedelta(days=prediction)#Prediction here contains the number of predicted days 
    #We get the day (For example Saturday, Monday...)
    week_day_discharge=date_discharge_hopital.day_name()
    #We get the month(For example January, March...)
    discharge_month = date_discharge_hopital.month_name()
    #We get the year of discharge from the hospital (for example 2021)
    discharge_year = date_discharge_hopital.year
    #We get the day of discharge from the hospital (For example 11, 25, 30)
    discharge_date_day =  date_discharge_hopital.day
    #We prepare the return string containing the prediction of the release date
    string_prediction=" is likely to be discharged from the hospital on "+str(week_day_discharge) + ", " +str(discharge_date_day) + " "+ str(discharge_month) + " "+ str(discharge_year) 
    #We return the index.html page with the formatted result
    #N.B: prediction_text will be called in the index.html page to return the result
    return render_template('index.html',prediction_text='This Patient {}'.format(string_prediction))
#We execute our Flask application 
if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    
