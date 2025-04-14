import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("LabelledData.csv")

# Disease-condition mapping
condition_weights = {
    'diabetes': {'fibre_g': 0.15, 'protein_g': 0.12, 'freesugar_g': -0.15, 'carb_g': -0.12, 'fat_g': -0.08},
    'obesity': {'fibre_g': 0.15, 'protein_g': 0.12, 'freesugar_g': -0.12, 'fat_g': -0.10, 'carb_g': -0.08},
    'high_bp': {'sodium_mg': -0.18, 'potassium_mg': 0.12, 'magnesium_mg': 0.08, 'calcium_mg': 0.08},
    'low_bp': {'sodium_mg': 0.15, 'iron_mg': 0.10, 'protein_g': 0.08, 'fibre_g': -0.05}
}

condition_avoid = {
    'diabetes': ["Sugary drinks", "White bread", "Pastries", "Fried foods"],
    'obesity': ["Fast food", "Processed snacks", "Sugary beverages", "Refined carbs"],
    'high_bp': ["Salty snacks", "Pickles", "Canned soups", "Red meat"],
    'low_bp': ["Alcohol", "High-carb meals without protein", "Bananas", "High potassium fruits"]
}

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

label_map = {
    'obesity': 'Health_Label_Obesity',
    'high_bp': 'Health_Label_HighBP',
    'low_bp': 'Health_Label_LowBP',
    'diabetes': 'Health_Label_Diabetes'
}

def recommend_top_foods_by_cluster(df, condition, healthy_cluster_label=0, top_n_meals=10, top_n_sides=5, diet_preference="Non Veg"):
    condition = condition.lower()
    if condition not in condition_weights:
        raise ValueError(f"Condition '{condition}' not supported. Choose from: {list(condition_weights.keys())}")

    weights = condition_weights[condition]
    health_col = label_map[condition]

    # Filter only the healthy cluster
    df_filtered = df[df[health_col] == healthy_cluster_label].copy()
    if df_filtered.empty:
        print(f"No foods found in cluster {healthy_cluster_label}. Using entire dataset.")
        df_filtered = df.copy()

    df_filtered = df_filtered.fillna(0)

    # Filter veg/non-veg
    if "Type" in df.columns:
        if diet_preference == "Veg":
            df_filtered = df_filtered[df_filtered["Type"] == 1]

    # Apply blocklist
    blocklists = {
        'obesity': ["Classic italian pasta", "Potato canjee (Aloo canjee)", "Boiled rice (Uble chawal)", "Sweet corn soup", "Gingo"],
        'low_bp': ["Carrot cake (Gajar ka cake)", "Apple banana pie", "Cheese pizza"],
        'diabetes': ["Jellied sunshine fruit salad", "Small onion pickle", "Mutton seekh kebab", "Fermented bamboo shoot pickle (Mesu pickle)", "Boti kebab", "Mango raita (Aam ka raita)", "Potato raita (Aloo ka raita)"],
        'high_bp': ["Mutton seekh kebab", "Boti kebab", "Dry masala chops", "Pasta hot pot", "Penne platter", "Cheese pizza"]
    }
    blocklist = blocklists.get(condition, [])
    df_filtered = df_filtered[~df_filtered['food_name'].isin(blocklist)]

    # Calculate score
    df_filtered.loc[:, 'score'] = sum(
        df_filtered[nutrient] * weight
        for nutrient, weight in weights.items()
        if nutrient in df_filtered.columns
    )

    # Separate meals and sides
    complete_meals_df = df_filtered[df_filtered['complete_meal.1'] == 1]
    side_dishes_df = df_filtered[df_filtered['complete_meal.1'] == 0]

    top_meals = complete_meals_df.sort_values(by='score', ascending=False).head(top_n_meals)['food_name'].tolist()
    top_sides = side_dishes_df.sort_values(by='score', ascending=False).head(top_n_sides)['food_name'].tolist()

    return top_meals, top_sides

# Streamlit UI
st.set_page_config(page_title="Smart Food Recommender", layout="wide")
st.title("🥗 Smart Health-Based Food Recommendation System")

# Age range as radio buttons
age_group = st.radio("Select Age Group", ["10-17", "18-35", "36-50", "51-65", "66-90"])
gender = st.radio("Select Gender", ["Male", "Female"])
condition = st.selectbox("Select Health Condition", ["", "Diabetes", "Obesity", "High BP", "Low BP"])
diet_preference = st.radio("Diet Preference", ["Veg", "Non Veg"])

# Ensure all inputs are provided
if condition and age_group and gender and diet_preference:
    condition_key = condition.lower().replace(" ", "_")
    st.success(f"🎯 Based on Age Group: {age_group}, Gender: {gender}, Condition: {condition}, Preference: {diet_preference}")

    top_meals, top_sides = recommend_top_foods_by_cluster(df, condition_key, diet_preference=diet_preference)

    if not top_meals and not top_sides:
        st.warning("⚠ No suitable food recommendations found. Try adjusting your condition or diet preference.")
    else:
        st.subheader("🍽 Top 10 Complete Meals")
        for i, meal in enumerate(top_meals):
            st.markdown(f"{i+1}. {meal}")

        st.subheader("🥗 Top 5 Side Dishes")
        for i, side in enumerate(top_sides):
            st.markdown(f"{i+1}. {side}")

        st.subheader("🧠 Key Nutrient Benefits")
        for nutrient in list(condition_weights[condition_key].keys())[:3]:
            if nutrient in nutrient_info:
                st.markdown(f"**{nutrient}**: {nutrient_info[nutrient]}")

        st.subheader("📛 Foods to Avoid")
        avoid_list = condition_avoid.get(condition_key, [])
        st.markdown("Avoid consuming:")
        st.markdown("- " + "\n- ".join(avoid_list))

        st.subheader("📥 Export Your Recommendations")
        recommendations_df = pd.DataFrame({
            "Complete Meals": pd.Series(top_meals),
            "Side Dishes": pd.Series(top_sides)
        })
        csv = recommendations_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Recommendations as CSV", data=csv, file_name="recommended_foods.csv", mime="text/csv")
else:
    st.info("Please select all the options (Age Group, Gender, Condition, and Diet Preference) to view recommendations.")

st.markdown("---")
