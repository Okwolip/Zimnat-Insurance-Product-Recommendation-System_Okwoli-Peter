import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

@st.cache_resource
def load_model():
    return joblib.load("zimnat_multi_label.pkl")

models = load_model()

product_cols = [
 'P5DA','RIBP','8NN1','7POT','66FJ','GYSR','SOP4',
 'RVSZ','PYUQ','LJR9','N2MW','AHXO','BSTQ','FM3X',
 'K6QO','QBOL','JWFN','JZ9D','J9JW','GHYX','ECY3'
]

st.title("Zimnat Insurance Product Recommendation System")
st.write("Enter customer details to get top 5 recommended insurance products.")

age = st.number_input("Customer Age", 18, 100)
days_since_join = st.number_input("Days Since Customer Joined", 0)
num_products = st.number_input("Number of Existing Products", 0)

branch_freq = st.slider("Branch Frequency Score", 0.0, 1.0)
occupation_freq = st.slider("Occupation Frequency Score", 0.0, 1.0)
occupation_cat_freq = st.slider("Occupation Category Frequency", 0.0, 1.0)

sex_enc = st.selectbox("Gender (Encoded)", [0, 1])
marital_enc = st.selectbox("Marital Status (Encoded)", [0, 1, 2])

cooc_sum = st.number_input("Product Co-occurrence Sum", 0)
cooc_max = st.number_input("Product Co-occurrence Max", 0)

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

if st.button("Recommend Products"):
    probs = np.array([
        models[label].predict_proba(input_data)[:, 1][0]
        for label in product_cols
    ])

top5_idx = np.argsort(-probs)[:5]
top5 = [(product_cols[i], probs[i]) for i in top5_idx]

st.subheader("Top 5 Recommended Products")
for p, pr in top5:
    st.write(f"{p}: {pr:.2%}")


labels = [p for p, _ in top5][::-1]
values = [v * 100 for _, v in top5][::-1]

fig, ax = plt.subplots()
ax.barh(labels, values)
ax.set_xlabel("Probability (%)")
ax.set_title("Top 5 Insurance Recommendations")

st.pyplot(fig)


