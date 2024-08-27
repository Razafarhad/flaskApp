from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model
model_filename = 'random_forest_model.pkl'
model = joblib.load(model_filename)
print(f"Model loaded from {model_filename}")

# Load the dataset to be used for recommendations
file_path = 'dataset.xlsx'  # Replace with your actual file path
data = pd.read_excel(file_path)
data = data.dropna()

# Encode 'Animal Type' and preserve original labels
data['Animal Type'] = data['Animal Type'].astype('category')
animal_type_labels = data['Animal Type'].cat.categories
data['Animal Type'] = data['Animal Type'].cat.codes

# Function to get top N recommendations based on budget, age, and pet type preference
def get_top_n_recommendations(data, budget, age, pet_type_input, n=5):
    # Filter data based on budget and age
    filtered_data = data[(data['Budget (PKR)'] <= budget) & (data['Animal Age (months)'] <= age)]

    # If pet types are provided, filter data to include only the preferred pet types
    if pet_type_input:
        pet_types = [ptype.strip() for ptype in pet_type_input.split(',')]
        invalid_types = [ptype for ptype in pet_types if ptype not in animal_type_labels]

        if invalid_types:
            return {"error": f"Invalid pet types: {', '.join(invalid_types)}. Available options are: {', '.join(animal_type_labels)}"}

        # Encode pet types for filtering
        pet_type_encoded = [animal_type_labels.get_loc(ptype) for ptype in pet_types]
        filtered_data = filtered_data[filtered_data['Animal Type'].isin(pet_type_encoded)]

    # Get the top N records from the filtered data
    top_n_recommendations = filtered_data.head(n)

    return top_n_recommendations

@app.route('/predict', methods=['POST'])
def predict():
    try:
        req_data = request.json
        budget = float(req_data.get('budget'))
        age = float(req_data.get('age'))
        pet_type_input = req_data.get('pet_type')
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid input"}), 400

    top_n_recommendations = get_top_n_recommendations(data, budget, age, pet_type_input, n=5)

    if isinstance(top_n_recommendations, dict) and 'error' in top_n_recommendations:
        return jsonify(top_n_recommendations), 400

    if not top_n_recommendations.empty:
        recommendations = [
            {
                "Index": row.name,
                "Budget": row['Budget (PKR)'],
                "Age": row['Animal Age (months)'],
                "Pet Type": animal_type_labels[row['Animal Type']]
            }
            for index, row in top_n_recommendations.iterrows()
        ]
        return jsonify(recommendations)
    else:
        return jsonify({"error": "No recommendations found based on your preferences."}), 404

if __name__ == '__main__':
    app.run(debug=True)
