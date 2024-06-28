
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)


final_svm_model = joblib.load('final_svm_model.pkl')
final_nb_model = joblib.load('final_nb_model.pkl')
final_rf_model = joblib.load('final_rf_model.pkl')


data = pd.read_csv("Training.csv").dropna(axis=1)


symptoms = data.columns[:-1]


symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom.lower()] = index


encoder = LabelEncoder()
encoder.fit(data['prognosis'])
np.save('encoder_classes.npy', encoder.classes_)

data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}

def predictDisease(symptoms):
    symptoms = symptoms.lower().split(",")
    
    
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"].get(symptom.strip(), None)
        if index is not None:
            input_data[index] = 1
        else:
            print(f"Warning: Symptom '{symptom}' not found in the symptom index.")

    
    input_data = np.array(input_data).reshape(1, -1)
    
  
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
    
   
    final_prediction = max(set([rf_prediction, nb_prediction, svm_prediction]), key=list([rf_prediction, nb_prediction, svm_prediction]).count)
    
    return final_prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        symptoms = request.form.get('symptoms')
        if not symptoms:
            return jsonify({'error': 'Please enter the symptoms'}), 400
        predicted_disease = predictDisease(symptoms)
        return jsonify({'disease': predicted_disease})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
