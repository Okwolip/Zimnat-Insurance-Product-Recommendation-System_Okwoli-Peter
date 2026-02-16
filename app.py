import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Zimnat Insurance Recommender",
    page_icon="💼",
    layout="wide"
)


# LOAD MODEL

@st.cache_resource
def load_model():
    return joblib.load("zimnat_multi_label.pkl")

models = load_model()


# PRODUCT LIST (Must match training order)

product_cols = [
    'P5DA','RIBP','8NN1','7POT','66FJ','GYSR','SOP4',
    'RVSZ','PYUQ','LJR9','N2MW','AHXO','BSTQ','FM3X',
    'K6QO','QBOL','JWFN','JZ9D','J9JW','GHYX','ECY3'
]


# HEADER

st.title("💼 Zimnat Insurance Product Recommendation System")
st.markdown("""
This intelligent system analyzes customer characteristics and recommends  
the **Top 5 most suitable insurance products** using machine learning.

Please complete all required fields accurately.
""")

st.divider()


# CUSTOMER INFORMATION SECTION

st.header("📋 Customer Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Customer Age", min_value=18, max_value=100)
    days_since_join = st.number_input("Days Since Customer Joined", min_value=0)
    num_products = st.number_input("Number of Existing Products", min_value=0)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

# Internal encoding
sex_enc = 0 if gender == "Male" else 1

marital_mapping = {
    "Single": 0,
    "Married": 1,
    "Divorced": 2
}
marital_enc = marital_mapping[marital_status]

st.divider()


# BRANCH & OCCUPATION DATA

st.header("🏢 Branch & Occupation Metrics")

col3, col4 = st.columns(2)

with col3:
    branch_freq = st.slider(
        "Branch Popularity Score",
        0.0, 1.0,
        help="Represents how frequently products are purchased in this branch."
    )
    occupation_freq = st.slider(
        "Occupation Popularity Score",
        0.0, 1.0,
        help="Represents product uptake frequency for this occupation."
    )

with col4:
    occupation_cat_freq = st.slider(
        "Occupation Category Score",
        0.0, 1.0,
        help="Represents insurance engagement level of this occupation group."
    )

st.divider()

# PRODUCT HISTORY SECTION

st.header("📦 Product Relationship Metrics")

cooc_sum = st.number_input(
    "Product Co-occurrence Sum",
    min_value=0,
    help="Total number of related insurance products historically purchased together."
)

cooc_max = st.number_input(
    "Product Co-occurrence Max",
    min_value=0,
    help="Highest frequency of co-purchase between any two products."
)

st.divider()


# VALIDATION FUNCTION

def validate_inputs():
    if age < 18:
        return False
    if days_since_join < 0:
        return False
    if num_products < 0:
        return False
    return True


# PREDICTION BUTTON

if st.button("🔍 Generate Recommendations"):

    if not validate_inputs():
        st.error("⚠️ Please ensure all inputs are valid.")
    else:

        with st.spinner("Analyzing customer profile and generating recommendations..."):

            try:
                # Prepare model input
                input_data = pd.DataFrame([{
                    'age': age,
                    'days_since_join': days_since_join,
                    'num_products': num_products,
                    'branch_code_freq': branch_freq,
                    'occupation_code_freq': occupation_freq,
                    'occupation_category_code_freq': occupation_cat_freq,
                    'sex_enc': sex_enc,
                    'marital_status_enc': marital_enc,
                    'cooc_sum': cooc_sum,
                    'cooc_max': cooc_max
                }])

                # Generate probabilities
                probs = np.array([
                    models[label].predict_proba(input_data)[:, 1][0]
                    for label in product_cols
                ])

                # Get Top 5
                top5_idx = np.argsort(-probs)[:5]
                top5 = [(product_cols[i], probs[i]) for i in top5_idx]

                st.success("✅ Recommendation Generated Successfully")

                st.subheader("🏆 Top 5 Recommended Products")

                # Display results
                for product, probability in top5:
                    st.write(f"**{product}** — {probability:.2%} likelihood")

                # Plot chart
                labels = [p for p, _ in top5][::-1]
                values = [v * 100 for _, v in top5][::-1]

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(labels, values)
                ax.set_xlabel("Predicted Probability (%)")
                ax.set_title("Top 5 Insurance Recommendations")

                st.pyplot(fig)

                st.divider()

                # Confidence Explanation
                st.subheader("📊 Understanding the Confidence Scores")

                st.markdown("""
- Higher percentage indicates stronger predicted suitability.
- Scores above **70%** suggest strong recommendation confidence.
- Scores between **40% – 70%** indicate moderate likelihood.
- Scores below **40%** suggest lower predicted engagement probability.
                """)

            except Exception:
                st.error("⚠️ An unexpected error occurred during prediction. Please review inputs or contact system administrator.")
