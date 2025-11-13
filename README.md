# AI Food Advisor: Your Personal AI Nutritionist

## 1. Project Description
Navigating dietary choices with a health condition can be confusing and stressful, often leading to reliance on generic, impersonal advice.  
The **AI Food Advisor** tackles this challenge head-on, providing an instant, personalized, and data-driven conversational chatbot specifically designed to answer the critical question: **"Can I eat this?"**

This application moves beyond simple lookups with a powerful **"Two-Brain" AI architecture**:

- The first brain, a Conversational **"Receptionist,"** uses fuzzy string matching to understand natural language, typos, and phrasing variations effortlessly.  
- Once the user's food and health condition are identified, the second brain, an **"Expert" AI,** takes over. This system is powered by a set of highly accurate **XGBoost models**, trained on a detailed nutritional dataset. Instead of just retrieving a pre-written answer, it predicts whether a food is suitable based on its specific nutritional profile—analyzing crucial biomarkers like sugar, sodium, and fat content.

The result is an **evidence-based recommendation** with clear explanations, all within a seamless chat experience. The entire system runs **100% locally**, guaranteeing user privacy.

---

## 2. Problem Fit & Solution

**Problem:** Users struggle to know which foods are good or bad for their specific health conditions (diabetes, hypertension, etc.).  

**Solution:**  
We built an interactive chatbot that allows a user to:
- Input their condition and query a food item in a natural, conversational way.
- Receive a clear **"GOOD TO EAT"** or **"AVOID"** recommendation, predicted by a machine learning model based on nutritional data.
- Get a detailed explanation of why the recommendation was made, referencing nutritional biomarkers relevant to their condition.

This provides a **user-friendly, data-driven, and personalized** solution that empowers users to make smarter dietary decisions.

---

## 3. Features

- 💬 **Interactive Chat Interface:** Beautiful and intuitive UI for a natural conversational experience.  
- 🤖 **Intelligent Language Understanding:** Powered by *thefuzz*, handles typos, synonyms, and variations in user phrasing (e.g., "high bp" vs. "my blood pressure is high").  
- 📊 **Data-Driven Predictions:** Uses highly accurate **XGBoost models** trained on 11 nutritional features.  
- 🧾 **Dynamic Explanations:** AI generates explanations on the fly, referencing actual nutritional values (e.g., *"This dish has 25.4g of free sugar."*).  
- 📈 **Performance Metrics:** Training script evaluates models and prints accuracy reports.  
- 🔒 **100% Local & Private:** Runs fully on the user's machine with no external APIs.

---

## 4. AI/ML Tools and Architecture

### 🧠 The "Receptionist" (Conversational AI)
- **File:** `chatbot.py`  
- **Technology:** Python + `thefuzz`  
- **Role:** Extracts `food_item` and `condition` from the user’s input, even with typos or messy phrasing.

### 🧠 The "Expert" (Predictive AI)
- **Files:** `train_nutrition_model.py`, `nutrition_models.pkl`  
- **Technology:** Python + `pandas`, `scikit-learn`, `xgboost`  
- **Role:** Takes the clean data from the Receptionist, runs predictions using XGBoost models, and provides data for explanations.

---

## 5. Setup and Installation

### 📋 Prerequisites
- Python **3.8+**  
Check your version with:
```bash
python --version
```

### ⚙️ Step-by-Step Setup Instructions

#### Step 1: Clone or Download the Project
Clone the repository:
```bash
git clone <your-repo-link>
cd <project-folder>
```

Or download the ZIP and extract it. Navigate into the **project root folder** (where `main.py` is).

#### Step 2: Set Up a Virtual Environment (Recommended)
Create the environment:
```bash
python -m venv venv
```

Activate the environment:  
- **Windows (CMD/PowerShell):**
```bash
venv\Scripts\activate
```
- **macOS/Linux (bash/zsh):**
```bash
source venv/bin/activate
```

#### Step 3: Install Dependencies
Install all required libraries:
```bash
pip install -r requirements.txt
```

---

## 6. How to Run the Application


### Step 1: Start the Backend Server
```bash
python main.py
```
The server will start (default: `http://127.0.0.1:5000`). Keep this terminal window open.

### Step 2: Open the Chat Interface
- Navigate to the project folder.  
- Double-click `index.html` to open it in your browser.  
- Start chatting with your AI Nutritionist!

---
#### Optional: Train the AI Models (One-Time Task)
__trained model is already present for your convenience__

```bash
python train_nutrition_model.py
```
This creates `nutrition_models.pkl`. You only need to do this once.


## 📌 Notes
- Always keep your virtual environment active while running the project.  
- Close the terminal running the server to shut down the backend.

---

## ✅ Done!  
You now have a fully working **AI Food Advisor** chatbot that helps answer: *"Can I eat this?"*


### Demo Vedion link ;:-
https://drive.google.com/file/d/1adla_cLoOZfjyBUhgoDPCcfwrXa-IUVU/view?usp=sharing
