import pandas as pd
from thefuzz import process, fuzz

# This chatbot is upgraded to handle multiple health conditions.

# --- Data Loading & Synonym Mapping ---
try:
    df = pd.read_csv('food_nutrition_data.csv')
    KNOWN_FOODS = list(df['Dish Name'].str.lower().unique())
    
    ALL_CONDITIONS_WITH_ALIASES = {
        "diabetes": ["diabetes", "diabetic", "sugar problem", "blood sugar"],
        "hypertension": ["hypertension", "high blood pressure", "high bp", "bp issue", "blood pressure"],
        "hyperlipide": ["hyperlipide", "cholesterol", "high cholesterol"],
        "thyroid": ["thyroid", "thyroid disorder"],
        "obesity": ["obesity", "overweight", "weight issue", "weight loss"]
    }
    
    ALIAS_TO_CANONICAL = {alias: canonical for canonical, aliases in ALL_CONDITIONS_WITH_ALIASES.items() for alias in aliases}
    ALL_ALIASES = list(ALIAS_TO_CANONICAL.keys())
    print("Chatbot loaded known foods and expanded condition aliases.")
except FileNotFoundError:
    print("WARNING: food_nutrition_data.csv not found.")
    KNOWN_FOODS = []
    ALL_ALIASES = []
    ALIAS_TO_CANONICAL = {}

# --- State Management (Now supports a list of conditions) ---
conversation_state = {
    "food_item": None,
    "conditions": [] # Changed to a list
}

def reset_conversation():
    """Clears the chatbot's memory for a new query."""
    global conversation_state
    conversation_state = {"food_item": None, "conditions": []}
    print("Conversation state reset.")

def get_chatbot_response(user_message):
    """
    Processes a user's message, identifies a food and MULTIPLE conditions.
    """
    global conversation_state
    user_message = user_message.lower().strip()
    
    # --- 1. Intelligent Entity Extraction for Multiple Entities ---
    CONFIDENCE_THRESHOLD = 80

    # Extract food (still one per query for simplicity)
    if not conversation_state['food_item']:
        best_food_match = process.extractOne(user_message, KNOWN_FOODS, scorer=fuzz.partial_ratio)
        if best_food_match and best_food_match[1] > CONFIDENCE_THRESHOLD:
            conversation_state['food_item'] = best_food_match[0]

    # Extract ALL matching conditions
    # THE FIX: The function is called 'extract', not 'extractAll'.
    best_condition_matches = process.extract(user_message, ALL_ALIASES, scorer=fuzz.partial_ratio)
    for match, score in best_condition_matches:
        if score > CONFIDENCE_THRESHOLD:
            canonical_condition = ALIAS_TO_CANONICAL.get(match)
            # Add to list if it's not already there
            if canonical_condition and canonical_condition not in conversation_state['conditions']:
                conversation_state['conditions'].append(canonical_condition)
    
    # --- 2. Conversational Logic ---
    food = conversation_state['food_item']
    conditions = conversation_state['conditions']

    if food and conditions:
        # SUCCESS: We have everything we need.
        conditions_str = " and ".join([c.capitalize() for c in conditions])
        response_text = f"Okay, I'm checking if **{food.title()}** is suitable for **{conditions_str}**."
        extracted_info = conversation_state.copy()
        return response_text, extracted_info
    
    elif food and not conditions:
        return f"Got it, you're asking about **{food.title()}**. To give you the best advice, could you please tell me your health condition(s)?", None
        
    elif conditions and not food:
        conditions_str = ", ".join([c.capitalize() for c in conditions])
        return f"Okay, for your condition(s) (**{conditions_str}**), what food are you curious about?", None
        
    else:
        return "Hello! I can help with food recommendations for your health. What food and condition(s) are you thinking of?", None
