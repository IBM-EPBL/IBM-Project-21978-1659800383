import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,confusion_matrix
import pickle
import feature
import sys
import joblib
import logging
app = Flask(__name__)

model = pickle.load(open('./Phishing_Websites.pkl', 'rb'))
@app.route('/')   #decorator
def phishing_detection():
    return render_template('index.html')


@app.route('/y_predict',methods=['POST'])
def y_predict():
    url = request.form['url']
    checkprediction = feature.main(url)
    prediction = model.predict(checkprediction)
    print(prediction)
    output=prediction[0]
    if(output==1):
        pred="your website is legitimate."
    elif(output==-1):
        pred="Wrong Website"
    

    return render_template('./index.html',prediction_text='{}'.format(pred),url=url)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.y_predict([np.array(list(data.values()))])

    output= prediction[0]
    return jsonify(output)







# if __name__ == '__main__':
#     app.debug=True
#     app.run(host='0.0.0.0', port=5000)
# @app.route('/predict/', methods=['GET','POST'])
# @login_required
# def predict():
#     if request.method == 'POST':
#         url = request.form['url']
#         checkprediction = inputScript.main(url)
#         print(url)
#         print(checkprediction)
#         prediction = model.predict(checkprediction)
#         print(prediction)
#         output=prediction[0]
#         if(output==1):
#             pred="Safe,legitimate link"
            
#         else:
#             pred="Malicious URL alert!"
#         if(session and session['logged_in']):
#             if(session['logged_in']==True):
#                 return render_template('index.html',userInfo=session['user'],pred=pred)
#             else:
#                 return render_template('./templates/prediction-result.html',pred=pred)
#         else:
#             return render_template('./templates/prediction-result.html',pred=pred)
#     elif request.method == 'GET':
#         return render_template('index.html',userInfo=session['user'])
if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)

