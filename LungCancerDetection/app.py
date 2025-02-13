from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from joblib import load
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

# Initialize Flask app
app = Flask(__name__)

# Load the trained Random Forest model
model = load(r'C:\Users\rpkja\LungCancerDetection\Random Forest_model.pkl')

# Define ordinal mappings (for encoding categorical inputs)
ordinal_mappings = {
    "Air Pollution": {
        "Minimal Exposure": 0,
        "Very Low Exposure": 1,
        "Low Exposure": 2,
        "Moderate-Low Exposure": 3,
        "Moderate Exposure": 4,
        "Moderate-High Exposure": 5,
        "High Exposure": 6,
        "Extreme Exposure": 7,
    },
    "Alcohol Usage": {
        "Not Used": 0,
        "Very Low Usage": 1,
        "Low Usage": 2,
        "Moderate-Low Usage": 3,
        "Moderate Usage": 4,
        "Moderate-High Usage": 5,
        "High Usage": 6,
        "Excessive Usage": 7,
    },
    "Genetic Risk": {
        "No Risk": 0,
        "Minimal Risk": 1,
        "Very Low Risk": 2,
        "Low Risk": 3,
        "Moderate Risk": 4,
        "High Risk": 5,
        "Elevated Risk": 6,
    },
    "Lung Disease": {
        "Healthy Lungs": 0,
        "Occasional Mild Cough": 1,
        "Seasonal Allergies,Mild Asthma": 2,
        "Moderate Asthma": 3,
        "Moderate to Significant Symptoms": 4,
        "Significant Symptoms": 5,
        "Severe Symptoms": 6,
    },
    "Obesity": {
        "Normal Weight": 0,
        "Slightly Over Weight": 1,
        "At Risk of Obesity": 2,
        "Pre-Obesity": 3,
        "Class 1 Obesity": 4,
        "Class 2 Obesity": 5,
        "Class 3 Obesity": 6,
    },
    "Smoking": {
        "Non-Smoker": 0,
        "Passive Smoker": 1,
        "Minimal Smoking": 2,
        "Very Low Smoking": 3,
        "Low Smoking": 4,
        "Moderate-Low Smoking": 5,
        "Moderate Smoking": 6,
        "Moderate-High Smoking": 7,
        "High Smoking": 8,
    },
}

# Define function to encode categorical values
def encode_input(data):
    encoded_data = []

    for feature, value in data.items():
        if feature in ordinal_mappings:
            encoded_value = ordinal_mappings[feature].get(value, -1)
            encoded_data.append(int(encoded_value))  # Convert to integer
        else:
            try:
                # Convert numeric values (like Age) to int or float
                if feature == "Age":  # Ensure Age is handled as a numeric value
                    encoded_data.append(int(value))
                else:
                    encoded_data.append(float(value))  # Convert other numeric features
            except ValueError:
                # If there's an error converting, handle it gracefully
                encoded_data.append(0)  # Default value for invalid inputs

    return np.array(encoded_data).reshape(1, -1)

# Define LIME explainer
def explain_prediction(model, input_data, training_data):
    # Create the LIME explainer
    explainer = LimeTabularExplainer(
        training_data=training_data,
        mode="classification",
        feature_names=["Age", "Air Pollution", "Alcohol Usage", "Genetic Risk", "Lung Disease", "Obesity", "Smoking"],
        discretize_continuous=True
    )

    # Generate explanation
    explanation = explainer.explain_instance(input_data[0], model.predict_proba)

    # Convert explanation to a readable string format
    explanation_text = "Factors influencing the prediction:\n"
    for term, weight in explanation.as_list():
        # Split the term to extract the feature name
        feature_name = term.split('>')[0].strip()  # Remove the threshold part (e.g., "Smoking > 0.73")

        # Convert the weight into a percentage
        weight_percentage = round(weight * 100, 2)

        # Check if the contribution is positive or negative and adjust the text accordingly
        if weight >= 0:
            explanation_text += f"{feature_name} contributes to increase the risk by {weight_percentage}%.\n"
        else:
            explanation_text += f"{feature_name} contributes to decrease the risk by {abs(weight_percentage)}%.\n"

    return explanation_text

@app.route('/')
def home():
    return render_template('index.html')  # Serve the HTML frontend

@app.route('/submit', methods=['POST'])
def submit():
    # Get input data from the form
    user_data = {
        "Age": request.form['age'],
        "Air Pollution": request.form['Air Pollution'],
        "Alcohol Usage": request.form['Alcohol Usage'],
        "Genetic Risk": request.form['Genetic Risk'],
        "Lung Disease": request.form['Lung Disease'],
        "Obesity": request.form['Obesity'],
        "Smoking": request.form['Smoking']
    }

    # Encode the input data
    encoded_input = encode_input(user_data)

    # Define the severity classes mapping
    severity_mapping = {0: "Healthy", 1: "Low", 2: "Medium", 3: "High"}

    # Get prediction from the model
    prediction_value = model.predict(encoded_input)[0]

    # Map the prediction to the severity class
    severity_prediction = severity_mapping.get(prediction_value, "Unknown")

    # Dummy training data for LIME (in real case, you need to load actual training data)
    # Example: This should be your training dataset (which should be numerical)
    training_data = np.random.rand(100, 7)  # Placeholder (replace with actual data)

    # Generate explanation using LIME
    explanation = explain_prediction(model, encoded_input, training_data)

    # Prepare the result to return to the frontend
    result = {
        "prediction": severity_prediction,  # Display the predicted severity class
        "explanation": explanation  # Display the LIME explanation
    }

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)