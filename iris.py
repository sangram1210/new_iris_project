from flask import Flask,request,render_template,redirect,jsonify
import numpy as np
import pandas as pd
import pickle

with open('model.pkl','rb') as file:
    model = pickle.load(file)


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("iris.html")

@app.route("/iris",methods=["Post"])
def predict():
    SepalLengthCm = float(request.form['sl'])
    SepalWidthCm = float(request.form['sw'])
    PetalLengthCm = float(request.form['pl'])
    PetalWidthCm = float(request.form['pw'])

    data = np.array([SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm],ndmin=2)

    result = model.predict(data)

    if result[0]==0:
        pred="Iris-setosa"
    if result[0]==1:
        pred="Iris-versicolor"
    if result[0]==2:
        pred="Iris-virginica"

    return render_template("iris.html",prediction=pred)




if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080)