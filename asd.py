from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pandas as pd
from joblib import load
from category_encoders import BinaryEncoder
import os

# Load cancer prediction model
model = load("decision_tree_model.joblib")

# Load dataset for encoder fitting
data = pd.read_csv("dataset.csv")

# Define categorical features for encoding
categorical_features = [
    'Age', 'Tumor Size (cm)', 'Cost of Treatment (USD)', 
    'Economic Burden (Lost Workdays per Year)', 'Country', 'Gender', 'Tobacco Use', 'Alcohol Consumption',
    'HPV Infection', 'Betel Quid Use', 'Chronic Sun Exposure', 'Poor Oral Hygiene', 'Diet (Fruits & Vegetables Intake)',
    'Family History of Cancer', 'Compromised Immune System', 'Oral Lesions', 'Unexplained Bleeding', 
    'Difficulty Swallowing', 'White or Red Patches in Mouth', 'Treatment Type', 'Early Diagnosis'
]

# Fit encoder on the dataset
encoder = BinaryEncoder()
encoder.fit_transform(data[categorical_features])

# Initialize Flask app
api = Flask(__name__)
CORS(api)

# Cancer risk prediction endpoint
@api.route('/predict', methods=['POST'])
@cross_origin(origins='*')
def predict_cancer_risk():
    try:
        data = request.json['inputs']
        input_df = pd.DataFrame(data)

        # Encode categorical features
        input_encoded = encoder.transform(input_df[categorical_features])

        # Drop original categorical columns
        input_df = input_df.drop(categorical_features, axis=1)

        # Add dummy 'ID' field if needed
        input_df['ID'] = 0

        # Prepare final input for model
        input_encoded = input_encoded.reset_index(drop=True)
        input_df = input_df.reset_index(drop=True)
        final_input = pd.concat([input_df, input_encoded], axis=1)

        # Predict
        prediction = model.predict_proba(final_input)
        class_labels = model.classes_

        # Format response
        response = []
        for prob in prediction:
            prob_dict = {str(k): round(float(v) * 100, 2) for k, v in zip(class_labels, prob)}
            response.append(prob_dict)

        return jsonify({'prediction': response})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    api.run(debug=True, host='0.0.0.0', port=port)
