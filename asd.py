from flask import Flask, request, jsonify
from flask_cors import CORS
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

# Fit encoder on the dataset once (avoid fitting on every request)
encoder = BinaryEncoder()
encoder.fit(data[categorical_features])  # fit only, don't transform here

# Initialize Flask app
api = Flask(__name__)

# Enable CORS globally with default settings (allows all origins)
CORS(api)

@api.route('/predict', methods=['POST'])
def predict_cancer_risk():
    try:
        # Parse input JSON
        input_json = request.get_json()
        if not input_json or 'inputs' not in input_json:
            return jsonify({"error": "Missing 'inputs' in JSON body"}), 400
        
        data_list = input_json['inputs']

        # Convert list of dicts to DataFrame
        input_df = pd.DataFrame(data_list)

        # Check if all required categorical features exist
        missing_cols = [col for col in categorical_features if col not in input_df.columns]
        if missing_cols:
            return jsonify({"error": f"Missing required columns: {missing_cols}"}), 400

        # Encode categorical features
        encoded_features = encoder.transform(input_df[categorical_features])

        # Drop original categorical columns
        input_df_numeric = input_df.drop(columns=categorical_features)

        # Add dummy 'ID' column if your model requires it
        if 'ID' not in input_df_numeric.columns:
            input_df_numeric['ID'] = 0

        # Concatenate encoded categorical features with numeric columns
        final_input = pd.concat([input_df_numeric.reset_index(drop=True), encoded_features.reset_index(drop=True)], axis=1)

        # Predict probabilities with model
        prediction_probs = model.predict_proba(final_input)
        class_labels = model.classes_

        # Format response probabilities as percentages with two decimals
        response = []
        for probs in prediction_probs:
            prob_dict = {str(label): round(float(prob)*100, 2) for label, prob in zip(class_labels, probs)}
            response.append(prob_dict)

        return jsonify({'prediction': response})

    except Exception as e:
        # Log error and respond with error message
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    api.run(debug=True, host='0.0.0.0', port=port)
