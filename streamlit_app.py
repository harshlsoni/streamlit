import streamlit as st
import pandas as pd

# Load dataset
df = pd.read_csv("LabelledData.csv")

# Condition weights for scoring
condition_weights = {
    'diabetes': {'fibre_g': 0.15, 'protein_g': 0.12, 'freesugar_g': -0.15, 'carb_g': -0.12, 'fat_g': -0.08},
    'obesity': {'fibre_g': 0.15, 'protein_g': 0.12, 'freesugar_g': -0.12, 'fat_g': -0.10, 'carb_g': -0.08},
    'high_bp': {'sodium_mg': -0.18, 'potassium_mg': 0.12, 'magnesium_mg': 0.08, 'calcium_mg': 0.08},
    'low_bp': {'sodium_mg': 0.15, 'iron_mg': 0.10, 'protein_g': 0.08, 'fibre_g': -0.05}
}

# Foods to avoid
condition_avoid = {
    'diabetes': ["Sugary drinks", "White bread", "Pastries", "Fried foods"],
    'obesity': ["Fast food", "Processed snacks", "Sugary beverages", "Refined carbs"],
    'high_bp': ["Salty snacks", "Pickles", "Canned soups", "Red meat"],
    'low_bp': ["Alcohol", "High-carb meals without protein", "Bananas", "High potassium fruits"]
}

# Nutrient info for benefits section
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

# Label mapping
label_map = {
    'obesity': 'Health_Label_Obesity',
    'high_bp': 'Health_Label_HighBP',
    'low_bp': 'Health_Label_LowBP',
    'diabetes': 'Health_Label_Diabetes'
}

# Recommendation function
def recommend_top_foods_by_cluster(df, condition, healthy_cluster_label=0, top_n_meals=10, top_n_sides=5, diet_preference="Non Veg"):
    condition = condition.lower()
    weights = condition_weights[condition]
    health_col = label_map[condition]

    df_filtered = df[df[health_col] == healthy_cluster_label].copy()
    if df_filtered.empty:
        df_filtered = df.copy()

    df_filtered = df_filtered.fillna(0)

    if "Type" in df.columns:
        if diet_preference == "Veg":
            df_filtered = df_filtered[df_filtered["Type"] == 1]

    blocklists = {
        'obesity': ["Classic italian pasta","Meat stock", "Potato canjee (Aloo canjee)", "Boiled rice (Uble chawal)", "Sweet corn soup", "Gingo"],
        'low_bp': ["Carrot cake (Gajar ka cake)", "Apple banana pie", "Cheese pizza"],
        'diabetes': ["Jellied sunshine fruit salad", "Small onion pickle", "Mutton seekh kebab", "Fermented bamboo shoot pickle (Mesu pickle)", "Boti kebab", "Mango raita (Aam ka raita)", "Potato raita (Aloo ka raita)"],
        'high_bp': ["Mutton seekh kebab", "Boti kebab", "Dry masala chops", "Pasta hot pot", "Penne platter", "Cheese pizza"]
    }

    blocklist = blocklists.get(condition, [])
    df_filtered = df_filtered[~df_filtered['food_name'].isin(blocklist)]

    df_filtered['score'] = sum(
        df_filtered[nutrient] * weight
        for nutrient, weight in weights.items()
        if nutrient in df_filtered.columns
    )

    complete_meals_df = df_filtered[df_filtered['complete_meal.1'] == 1]
    side_dishes_df = df_filtered[df_filtered['complete_meal.1'] == 0]

    top_meals = complete_meals_df.sort_values(by='score', ascending=False).head(top_n_meals)['food_name'].tolist()
    top_sides = side_dishes_df.sort_values(by='score', ascending=False).head(top_n_sides)['food_name'].tolist()

    return top_meals, top_sides

# ----------------------- UI SECTION ------------------------

st.set_page_config(page_title="Smart Food Recommender", layout="wide")

st.markdown("<h1 style='text-align: center;'>🥗 Smart Health-Based Food Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: gray;'>Your Personalized Path to Better Health</p>", unsafe_allow_html=True)

st.divider()

condition = st.selectbox("Select Your Health Condition", ["", "Diabetes", "Obesity", "High BP", "Low BP"])
diet_preference = st.radio("Choose Your Diet Preference", ["Veg", "Non Veg"])

if condition and diet_preference:
    condition_key = condition.lower().replace(" ", "_")

    top_meals, top_sides = recommend_top_foods_by_cluster(df, condition_key, diet_preference=diet_preference)

    if not top_meals and not top_sides:
        st.warning("⚠ No suitable food recommendations found. Try adjusting your condition or diet preference.")
    else:
        st.divider()

        # 🍽 Top Meals & 🥗 Sides in 2 Columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🍽 Top 10 Complete Meals")
            st.caption("Satisfy Your Hunger and Your Health Goals!")
            for i, meal in enumerate(top_meals, 1):
                st.markdown(f"""
                <div style="background-color:#f5f5f5; padding:10px; border-radius:10px; margin-bottom:10px;">
                    <b>{i}. {meal}</b>
                </div>""", unsafe_allow_html=True)

        with col2:
            st.subheader("🥗 Top 5 Side Dishes")
            st.caption("Snack Smart, Stay Sharp!")
            for i, side in enumerate(top_sides, 1):
                st.markdown(f"""
                <div style="background-color:#f5f5f5; padding:10px; border-radius:10px; margin-bottom:10px;">
                    <b>{i}. {side}</b>
                </div>""", unsafe_allow_html=True)

        st.divider()

        st.subheader("🧠 Key Nutrient Benefits")
        for nutrient in list(condition_weights[condition_key].keys())[:3]:
            if nutrient in nutrient_info:
                st.markdown(f"🔸 *{nutrient}*: {nutrient_info[nutrient]}")

        st.divider()

        # "Avoid These" Section
        with st.expander("📛 Foods to Avoid"):
            avoid_list = condition_avoid.get(condition_key, [])
            st.markdown("- " + "\n- ".join(avoid_list))

        st.divider()

        # Download button
        recommendations_df = pd.DataFrame({
            "Complete Meals": pd.Series(top_meals),
            "Side Dishes": pd.Series(top_sides)
        })
        csv = recommendations_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Your Recommendations as CSV", data=csv, file_name="recommended_foods.csv", mime="text/csv")

else:
    st.info("Please select your Health Condition and Diet Preference to get recommendations.")

st.markdown("---")
