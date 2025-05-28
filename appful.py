from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import zipfile
import os

app = Flask(__name__)

def unzip_model(zip_path, extract_path, model_filename):
    """تفك ضغط ملف zip إذا لم يكن الملف الهدف موجود."""
    if not os.path.exists(model_filename):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract(model_filename, extract_path)
        print(f"Extracted {model_filename} from {zip_path}")

unzip_model("model.zip", ".", "model.pkl")
unzip_model("encoder.zip", ".", "encoder.pkl")
unzip_model("modeld.zip", ".", "modeld.pkl")
unzip_model("Duencoder.zip", ".", "Duencoder.pkl")

strategy_model = joblib.load("model.pkl")
strategy_encoders = joblib.load("encoder.pkl")

duration_model = joblib.load("modeld.pkl")
duration_encoders = joblib.load("Duencoder.pkl")

model3 = joblib.load("model3.pkl")
encoder3 = joblib.load("encoder3.pkl")

@app.route("/predict_strategy", methods=["POST"])
def predict_strategy():
    try:
        data = request.get_json()

        business_type = data["businessType"]
        target_age = data["targetAge"]
        main_channel = data["mainChannel"]
        marketing_objective = data["marketingObjective"]
        duration = int(data["duration"])
        budget = float(data["budget"])

        check_values = {
            "Business_Type": business_type,
            "Target_Age": target_age,
            "Main_Channel": main_channel,
            "Marketing_Objective": marketing_objective,
        }

        for key, value in check_values.items():
            if value not in strategy_encoders[key].classes_:
                return jsonify({"error": f"Invalid value '{value}' for {key}"}), 400

        input_df = pd.DataFrame([{
            "Business_Type": strategy_encoders["Business_Type"].transform([business_type])[0],
            "Target_Age": strategy_encoders["Target_Age"].transform([target_age])[0],
            "Budget": budget,
            "Main_Channel": strategy_encoders["Main_Channel"].transform([main_channel])[0],
            "Marketing_Objective": strategy_encoders["Marketing_Objective"].transform([marketing_objective])[0],
            "Campaign_Duration_Weeks": duration
        }])

        prediction = strategy_model.predict(input_df)
        strategy = strategy_encoders["Detailed_Marketing_Strategy"].inverse_transform(prediction)[0]

        return jsonify({"strategy": strategy})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict_duration", methods=["POST"])
def predict_duration():
    try:
        data = request.get_json()

        business_type = data["businessType"]
        budget = float(data["budget"])
        main_channel = data["mainChannel"]
        marketing_objective = data["marketingObjective"]

        check_values = {
            "Business_Type": business_type,
            "Main_Channel": main_channel,
            "Marketing_Objective": marketing_objective,
        }

        for key, value in check_values.items():
            if value not in duration_encoders[key].classes_:
                return jsonify({"error": f"Invalid value '{value}' for {key}"}), 400

        input_df = pd.DataFrame([[ 
            duration_encoders["Business_Type"].transform([business_type])[0],
            budget,
            duration_encoders["Main_Channel"].transform([main_channel])[0],
            duration_encoders["Marketing_Objective"].transform([marketing_objective])[0]
        ]], columns=["Business_Type", "Budget", "Main_Channel", "Marketing_Objective"])

        prediction = duration_model.predict(input_df)
        predicted_duration = int(prediction[0])

        return jsonify({"duration": predicted_duration})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received data:", data)

        platform = data['platform']                   
        duration_weeks = float(data['duration'])      
        budget = float(data['budget'])               
        ad_type = data['ad_type']                     
        previous_ctr = float(data['previous_ctr'])    
        previous_conversion_rate = float(data['conversion_rate'])  

        if platform not in encoder3['platform'].classes_:
            return jsonify({'error': f'Invalid platform: {platform}'}), 400
        if ad_type not in encoder3['ad_type'].classes_:
            return jsonify({'error': f'Invalid ad_type: {ad_type}'}), 400

        encoded_platform = encoder3['platform'].transform([platform])[0]
        encoded_ad_type = encoder3['ad_type'].transform([ad_type])[0]

        input_data = np.array([[encoded_platform, duration_weeks, budget, encoded_ad_type, previous_ctr, previous_conversion_rate]])

        prediction = model3.predict(input_data)[0]

        return jsonify({
            'prediction': int(prediction),
            'result': 'Successful' if prediction == 1 else 'Failed'
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
