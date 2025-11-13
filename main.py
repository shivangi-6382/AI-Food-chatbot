from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
from chatbot import get_chatbot_response, reset_conversation

app = Flask(__name__)
CORS(app)

# --- Load the AI Models and Data at Startup ---
try:
    models = joblib.load('nutrition_models.pkl')
    df = pd.read_csv('food_nutrition_data.csv')
    df.set_index('Dish Name', inplace=True)
    print("Expert models and nutritional data loaded successfully.")
except FileNotFoundError:
    print("\n--- FATAL ERROR ---")
    print("Model/data files not found. Please run 'train_nutrition_model.py' first.")
    models = None
    df = None

@app.route('/chat', methods=['POST'])
def chat():
    if not models or df is None:
        return jsonify({"error": "Models or data not loaded. Please run the training script."}), 500
        
    data = request.json
    user_message = data.get('message', '')

    response_text, extracted_info = get_chatbot_response(user_message)

    if extracted_info:
        food = extracted_info['food_item']
        conditions = extracted_info['conditions']
        
        # --- MAJOR UPGRADE: Loop through each condition and generate a report ---
        analysis_parts = []
        
        try:
            food_features_df = df.loc[food.title()].drop(
                ['diabetes', 'hypertension', 'hyperlipide', 'thyroid']
            )
            food_features = food_features_df.values.tolist()

            for condition in conditions:
                if condition in models:
                    expert_model = models[condition]
                    prediction = expert_model.predict([food_features])[0]
                    result_text = "GOOD TO EAT" if prediction == 1 else "AVOID"
                    
                    explanation = ""
                    if condition == 'diabetes':
                        sugar_val = food_features_df['Free Sugar']
                        explanation = f"It has {sugar_val:.2f}g of free sugar."
                    elif condition == 'hypertension':
                        sodium_val = food_features_df['Sodium']
                        explanation = f"It contains {sodium_val:.2f}mg of sodium."
                    elif condition == 'hyperlipide':
                        fats_val = food_features_df['Fats']
                        explanation = f"It contains {fats_val:.2f}g of fat."
                    elif condition == 'thyroid':
                        protein_val = food_features_df['Protein']
                        explanation = f"It has {protein_val:.2f}g of protein."
                    
                    # Create a formatted string for each condition's analysis
                    analysis_parts.append(
                        f"**For {condition.capitalize()}: {result_text}**\n_{explanation}_"
                    )

            # Assemble the final response
            if analysis_parts:
                full_analysis = "\n\n".join(analysis_parts)
                final_response = (
                    f"{response_text}\n\n"
                    f"### Nutritional Analysis for {food.title()}\n"
                    f"{full_analysis}"
                    f"\n\n*Disclaimer: This is AI-generated advice. Please consult a healthcare professional.*"
                )
            else:
                 final_response = "I couldn't analyze the conditions you mentioned. Please try again."

        except KeyError:
            final_response = f"Sorry, I couldn't find the nutritional details for '{food}' in my data."

        reset_conversation()
        return jsonify({'response': final_response})
    else:
        return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(port=5000, debug=True)

