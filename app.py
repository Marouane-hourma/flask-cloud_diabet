import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.ensemble import RandomForestClassifier  # Importez RandomForestClassif

app = Flask(__name__)
model = pickle.load(open('model_reg.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # Récupérer les valeurs du formulaire
    features = [float(x) for x in request.form.values()]
    # Convertir en un tableau numpy
    features_array = np.array(features).reshape(1, -1)
    # Faire la prédiction
    prediction = model.predict(features_array)
    # Récupérer le résultat de la prédiction
    if prediction == 1:
        output = "La personne est malade."
    else:
        output = "La personne n'est pas malade."

    return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)
