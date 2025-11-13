import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# This is the final, most robust training script. It handles class imbalance,
# which fixes the errors and makes the models more accurate.

print("Loading expanded nutritional data from food_nutrition_data.csv...")
try:
    df = pd.read_csv('food_nutrition_data.csv')
    print("CSV loaded successfully. Total records:", len(df))
except FileNotFoundError:
    print("\n--- FATAL ERROR ---")
    print("food_nutrition_data.csv not found. Please make sure the file is in the correct folder.")
    print("-------------------\n")
    exit()

# --- 1. Define Features (Nutrients) and Targets (Diseases) ---
features = [
    'Calories', 'Carbohydrate', 'Protein', 'Fats', 'Free Sugar', 'Fibre',
    'Sodium', 'Calcium', 'Iron', 'Vitamin C', 'Folate'
]
diseases = ['diabetes', 'hypertension', 'hyperlipide', 'thyroid']

# --- 2. Evaluate Model Performance ---
print("\n--- Evaluating Model Performance on a Test Set ---")
for disease in diseases:
    print(f"\n--- Metrics for: {disease.capitalize()} ---")
    
    X = df[features]
    y = df[disease]
    
    # Data Quality Check: Ensure there are both 0s and 1s in the data
    if y.nunique() < 2:
        print(f"Skipping {disease.capitalize()}: The dataset contains only one class (all GOOD or all AVOID). Model cannot be trained.")
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # THE FIX: Calculate scale_pos_weight to handle class imbalance
    # This tells the model to pay more attention to the rare class (usually 'AVOID').
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1) if np.sum(y_train == 1) > 0 else 1
    
    # Initialize the XGBoost model WITH the fix and cleaned parameters
    model = xgb.XGBClassifier(
        objective='binary:logistic', 
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['AVOID (0)', 'GOOD (1)'], zero_division=0))

# --- 3. Train Final Models on the Entire Dataset ---
final_models = {}
print("\n--- Training Final Models on 100% of the Data ---")

for disease in diseases:
    print(f"Training final model for: {disease.capitalize()}...")
    
    X_full = df[features]
    y_full = df[disease]

    if y_full.nunique() < 2:
        print(f"Skipping final model for {disease.capitalize()}: Cannot train with only one class.")
        continue
    
    # Apply the same fix for the final models
    scale_pos_weight_full = np.sum(y_full == 0) / np.sum(y_full == 1) if np.sum(y_full == 1) > 0 else 1

    final_model = xgb.XGBClassifier(
        objective='binary:logistic', 
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight_full
    )
    final_model.fit(X_full, y_full)
    final_models[disease] = final_model
    print(f"Final model for {disease.capitalize()} is ready.")

# --- 4. Save Final Models ---
joblib.dump(final_models, 'nutrition_models.pkl')
print("\nAll final expert models have been trained and saved to nutrition_models.pkl")

