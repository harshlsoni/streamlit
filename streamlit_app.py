import streamlit as st
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(page_title="Smart Food Recommender", layout="wide")

# === 🍃 Custom CSS for Light Aesthetic & Background ===
st.markdown("""
<style>
/* General body styling */
body {
    background-image: url('https://images.unsplash.com/photo-1581891651010-2d27e1c7e3a9?auto=format&fit=crop&w=1950&q=80');
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
    font-family: 'Lora', serif;
    color: #333;
}

/* Overlay for better contrast */
body::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(255, 255, 255, 0.7);
    z-index: -1;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.9);
    border-right: 2px solid #ddd;
}

/* Header styling */
h1, h2, h3 {
    color: #2E8B57; /* Dark green for food theme */
    font-family: 'Montserrat', sans-serif;
}

/* Input widgets */
input, .stSlider, .stSelectbox, .stRadio, .stTextInput {
    background-color: rgba(255, 255, 255, 0.9) !important;
    border: 2px solid #76c7c0;
    color: #333;
    padding: 0.75rem;
    border-radius: 8px;
    font-family: 'Lora', serif;
}

/* Button styling */
.stButton > button {
    background-color: #FFA07A;
    color: white;
    border-radius: 12px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    border: none;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.stButton > button:hover {
    background-color: #FF8C00;
}

/* Slider and input fields hover */
.stSlider:hover, .stSelectbox:hover, .stRadio:hover, .stTextInput:hover {
    background-color: rgba(255, 255, 255, 1);
}

/* Title and subtitle text */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #3e6f48; /* Fresh green for headings */
}

/* Food list styling */
.top-list-box {
    background-color: rgba(255, 255, 255, 0.85);
    border-radius: 16px;
    padding: 2rem;
    margin-top: 2rem;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
}

.top-list-box h4 {
    color: #2E8B57;
    font-size: 1.2rem;
    font-weight: 600;
}

/* Card-like effect for lists */
ul {
    list-style-type: none;
}

ul li {
    background-color: #F0F8FF;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 10px;
    font-family: 'Lora', serif;
    border-left: 5px solid #76c7c0;
    transition: all 0.3s ease;
}

ul li:hover {
    background-color: #E0FFFF;
    transform: translateX(5px);
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
}

/* Footer text */
.stCaption {
    color: #2e2e2e;
    font-family: 'Arial', sans-serif;
    text-align: center;
    margin-top: 2rem;
}

/* Add custom fonts */
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;700&family=Montserrat:wght@600&display=swap');
</style>
""", unsafe_allow_html=True)

# === 📥 Load data ===
df = pd.read_csv("LabelledData.csv")

# === ⚖️ Condition-based weights ===
condition_weights = {
    'diabetes': {'fibre_g': 0.15, 'protein_g': 0.12, 'freesugar_g': -0.15, 'carb_g': -0.12, 'fat_g': -0.08},
    'obesity': {'fibre_g': 0.15, 'protein_g': 0.12, 'freesugar_g': -0.12, 'fat_g': -0.10, 'carb_g': -0.08},
    'high_bp': {'sodium_mg': -0.18, 'potassium_mg': 0.12, 'magnesium_mg': 0.08, 'calcium_mg': 0.08},
    'low_bp': {'sodium_mg': 0.15, 'iron_mg': 0.10, 'protein_g': 0.08, 'fibre_g': -0.05}
}

# 🛑 Foods to avoid
condition_avoid = {
    'diabetes': ["Sugary drinks", "White bread", "Pastries", "Fried foods"],
    'obesity': ["Fast food", "Processed snacks", "Sugary beverages", "Refined carbs"],
    'high_bp': ["Salty snacks", "Pickles", "Canned soups", "Red meat"],
    'low_bp': ["Alcohol", "High-carb meals without protein", "Bananas", "High potassium fruits"]
}

# 🧠 Nutrient info
nutrient_info = {
    'fibre_g': "Helps in digestion and regulates blood sugar.",
    'protein_g': "Essential for muscle repair and satiety.",
    'freesugar_g': "Should be limited to control sugar spikes.",
    'carb_g': "Primary energy source, but should be moderated.",
    'fat_g': "Necessary for hormones, but excess leads to weight gain.",
    'sodium_mg': "Excess sodium increases blood pressure.",
    'potassium_mg': "Helps reduce sodium effects and maintain BP.",
    'iron_mg': "Essential for blood production."
}

# 🔖 Label column mapping
label_map = {
    'obesity': 'Health_Label_Obesity',
    'high_bp': 'Health_Label_HighBP',
    'low_bp': 'Health_Label_LowBP',
    'diabetes': 'Health_Label_Diabetes'
}

# 🍽️ Food recommendation logic
def recommend_top_foods_by_cluster(df, condition, healthy_cluster_label=0, top_n_meals=10, top_n_sides=5):
    condition = condition.lower()
    weights = condition_weights[condition]
    health_col = label_map[condition]

    # Filter healthy cluster
    df_filtered = df[df[health_col] == healthy_cluster_label].copy()
    if df_filtered.empty:
        df_filtered = df.copy()

    df_filtered = df_filtered.fillna(0)

    blocklists = {
        'obesity': ["Classic italian pasta", "Potato canjee (Aloo canjee)", "Boiled rice (Uble chawal)", "Sweet corn soup", "Gingo"],
        'low_bp': ["Carrot cake (Gajar ka cake)", "Apple banana pie", "Cheese pizza"],
        'diabetes': ["Jellied sunshine fruit salad", "Small onion pickle", "Mutton seekh kebab", "Fermented bamboo shoot pickle (Mesu pickle)", "Boti kebab", "Mango raita (Aam ka raita)", "Potato raita (Aloo ka raita)"],
        'high_bp': ["Mutton seekh kebab", "Boti kebab", "Dry masala chops", "Pasta hot pot", "Penne platter", "Cheese pizza"]
    }
    blocklist = blocklists.get(condition, [])
    df_filtered = df_filtered[~df_filtered['food_name'].isin(blocklist)]

    df_filtered['score'] = sum(df_filtered[nutrient] * weight for nutrient, weight in weights.items() if nutrient in df_filtered.columns)

    meals = df_filtered[df_filtered['complete_meal.1'] == 1]
    sides = df_filtered[df_filtered['complete_meal.1'] == 0]

    top_meals = meals.sort_values(by='score', ascending=False).head(top_n_meals)['food_name'].tolist()
    top_sides = sides.sort_values(by='score', ascending=False).head(top_n_sides)['food_name'].tolist()

    return top_meals, top_sides

# === 🎯 UI Components ===
st.title("🥗 Smart Health-Based Food Recommendation System")
# === 🍃 Custom CSS for Light Aesthetic & Background ===
st.markdown("""
<style>
/* General body styling */
body {
    background-image: url('https://images.unsplash.com/photo-1581891651010-2d27e1c7e3a9?auto=format&fit=crop&w=1950&q=80');
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
    font-family: 'Lora', serif;
    color: #333;
    opacity: 0.85;  /* Adjusted transparency */
}

/* Overlay for better contrast */
body::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(255, 255, 255, 0.6);  /* Adjusted overlay opacity */
    z-index: -1;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.9);
    border-right: 2px solid #ddd;
}

/* Header styling */
h1, h2, h3 {
    color: #2E8B57; /* Dark green for food theme */
    font-family: 'Montserrat', sans-serif;
}

/* Input widgets */
input, .stSlider, .stSelectbox, .stRadio, .stTextInput {
    background-color: rgba(255, 255, 255, 0.9) !important;
    border: 2px solid #76c7c0;
    color: #333;
    padding: 0.75rem;
    border-radius: 8px;
    font-family: 'Lora', serif;
}

/* Slider background and handle fix */
div[data-baseweb="slider"] {
    background-color: transparent !important;
    padding: 0.5rem 0.5rem 1rem 0.5rem;
}

div[data-baseweb="slider"] > div {
    background: #76c7c0 !important;  /* Changed to a softer color */
    border-radius: 4px;
    height: 4px;
}

/* Slider handle */
div[data-baseweb="slider"] span[role="slider"] {
    border: 2px solid #76c7c0;
    box-shadow: 0 0 0 4px rgba(118, 199, 192, 0.3); /* Lighter green shadow */
}

/* Button styling */
.stButton > button {
    background-color: #FFA07A;
    color: white;
    border-radius: 12px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    border: none;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.stButton > button:hover {
    background-color: #FF8C00;
}

/* Food list styling */
.top-list-box {
    background-color: rgba(255, 255, 255, 0.85);
    border-radius: 16px;
    padding: 2rem;
    margin-top: 2rem;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
}

.top-list-box h4 {
    color: #2E8B57;
    font-size: 1.2rem;
    font-weight: 600;
}

/* Slider hover effect */
.stSlider:hover, .stSelectbox:hover, .stRadio:hover, .stTextInput:hover {
    background-color: rgba(255, 255, 255, 1);
}

/* Footer text */
.stCaption {
    color: #2e2e2e;
    font-family: 'Arial', sans-serif;
    text-align: center;
    margin-top: 2rem;
}

/* Add custom fonts */
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;700&family=Montserrat:wght@600&display=swap');
</style>
""", unsafe_allow_html=True)


# User input
age = st.slider("Select Age", 10, 90, 30)
gender = st.radio("Select Gender", ["Male", "Female"])
condition = st.selectbox("Select Health Condition", ["Diabetes", "Obesity", "High_BP", "Low_BP"])

if condition:
    condition_key = condition.lower()
    st.success(f"Based on your input (Age: {age}, Gender: {gender}, Condition: {condition}), here are your recommendations:")

    top_meals, top_sides = recommend_top_foods_by_cluster(df, condition_key)

    # 🍽️ Meals
    st.subheader("🍽️ Top 10 Complete Meals")
    for i, meal in enumerate(top_meals):
        st.markdown(f"**{i+1}.** {meal}")

    # 🥗 Sides
    st.subheader("🥗 Top 5 Side Dishes")
    for i, side in enumerate(top_sides):
        st.markdown(f"**{i+1}.** {side}")

    # 🧠 Tips
    st.subheader("🧠 Key Nutrient Benefits")
    for nutrient in list(condition_weights[condition_key].keys())[:3]:
        if nutrient in nutrient_info:
            st.markdown(f"🔹 **{nutrient.replace('_', ' ').title()}**: {nutrient_info[nutrient]}")

    # 🚫 Avoid
    st.subheader("📛 Foods to Avoid")
    st.markdown("- " + "\n- ".join(condition_avoid.get(condition_key, [])))

    # 💾 Download
    st.subheader("📥 Export Your Recommendations")
    recommendations_df = pd.DataFrame({
        "Complete Meals": pd.Series(top_meals),
        "Side Dishes": pd.Series(top_sides)
    })
    csv = recommendations_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download as CSV", data=csv, file_name="recommended_foods.csv", mime="text/csv")

st.markdown("---")
st.caption("Built with ❤️ for your health • Powered by Smart Food AI")
