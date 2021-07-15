'''from flask import Flask, render_template , request

import titanic

app = Flask(__name__,template_folder='template')


@app.route("/", methods = ['GET','POST'])
def index():
    if request.method == "POST":
        hrs = request.form['hrs']
        marks_pred = titanic.marks_prediction(hrs)
        print(marks_pred)
        
    return render_template('index.html')




@app.route("/sub", methods = ['POST'])
def submit():
    if request.method == "POST":
        name = request.form["username"]

    return render_template("sub.html", n = name )



if __name__ == "__main__":
    app.run(debug=True)'''


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__,template_folder='template')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)