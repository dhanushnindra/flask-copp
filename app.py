import pandas as pd
import numpy as np
from datetime import datetime
import itertools

# import sklearn.linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
oildataset = pd.read_csv('clean_total_dataset.csv')
oildataset.drop(['coronadate', 'datewotime','Close','Unnamed: 0'],axis=1,inplace=True)
print(oildataset.info())
file_name = "final_model.sav"
loaded_model = joblib.load(file_name)





# from flask import Flask,render_template,request
# app = Flask(__name__)

# @app.route('/')
# def hello_world():
#    return render_template('home.html')

# if __name__ == '__main__':
#    app.run()




print("SERVER running....")

import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import csv

app = Flask(__name__)
model = loaded_model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about/')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/dataset/')
def dataset():
    data = pd.read_csv('clean_total_dataset.csv')
    return render_template('dataset.html', tables=[data.to_html()], titles=[''])

@app.route('/predict',methods=['POST'])
def predict():
    #first convert all form input values into respective dtypes and into a Dframe
    for x in request.form.values():
        print(x)

    #date to ordinal value
    form_values_gen = request.form.values()
    form_values = list(itertools.islice(form_values_gen, 7))
    datetime_str = form_values[0]
    dateobject = datetime.strptime(datetime_str, '%Y-%m-%d').date()
    ordinalinput = dateobject.toordinal()
    print(form_values)

    print(ordinalinput)
    for x in form_values:
        print(x, type(x))
    
    Open_value = float(form_values[1])
    print(type(Open_value))

    High_value = float(form_values[2])
    print(type(High_value))

    Low_value = float(form_values[3])
    print(type(Low_value))


    Volume_value = int(form_values[4])

    Cases_value = int(form_values[5])

    Deaths_value = int(form_values[6])




    newrow = pd.DataFrame({
        'Open':Open_value,
        'High' : High_value,
        'Low' : Low_value,
        'Volume' : Volume_value,
        'ordinal_values' : ordinalinput,
        'cases' : Cases_value,
        'deaths' : Deaths_value}, index=[0])
    

    
    df = pd.concat([newrow, oildataset.loc[:1]]).reset_index(drop=True)
    pd.to_numeric(df['Open'])
    print(df.info())
    df= df[:1]
    print(df)

    data_pipeline = Pipeline([
        ('std_scaler', StandardScaler())
    ])
    data_pipeline.fit_transform(oildataset)

    df  = data_pipeline.transform(df)

    
    print(df)
    pred = loaded_model.predict(df)
    
    

    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)

    # output = round(prediction[0], 2)
    # return render_template('index.html')
    return render_template('index.html', prediction_text='Price could be $ {}'.format(pred))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
