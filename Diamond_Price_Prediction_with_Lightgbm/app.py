import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Diamond Input Form", layout="centered")

# Load model from model.pkl
@st.cache_resource  # cache so it loads only once [web:61][web:64]
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("Diamond Features Input Form")

# Use a form so all inputs are submitted together
with st.form("diamond_form"):
    st.subheader("Numeric features")

    col1, col2, col3 = st.columns(3)
    with col1:
        carat = st.number_input("Carat", min_value=0.0, step=0.01)
        depth = st.number_input("Depth", min_value=0.0, step=0.1)
        table = st.number_input("Table", min_value=0.0, step=0.1)
    with col2:
        x = st.number_input("x", min_value=0.0, step=0.01)
        y = st.number_input("y", min_value=0.0, step=0.01)
        z = st.number_input("z", min_value=0.0, step=0.01)
    with col3:
        st.markdown(" ")
        st.markdown("Enter dimensions in mm")

    st.markdown("---")

    st.subheader("Categorical features")

    cut = st.radio(
        "Cut",
        options=["Fair", "Good", "Very Good", "Premium", "Ideal"],
        horizontal=True,
    )

    color = st.radio(
        "Color",
        options=["D", "E", "F", "G", "H", "I", "J"],
        horizontal=True,
    )

    clarity = st.radio(
        "Clarity",
        options=["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
        horizontal=True,
    )

    submitted = st.form_submit_button("Submit")

# After submit: build feature vector and predict
if submitted:
    st.success("Form submitted successfully!")

    # Map categorical variables to the same encoding used during training
    cut_map = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
    color_map = {"D": 0, "E": 1, "F": 2, "G": 3, "H": 4, "I": 5, "J": 6}
    clarity_map = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4,
                   "VVS2": 5, "VVS1": 6, "IF": 7}

    features = np.array(
        [[
            carat,
            depth,
            table,
            x,
            y,
            z,
            cut_map[cut],
            color_map[color],
            clarity_map[clarity],
        ]]
    )  # shape (1, n_features) for sklearn predict [web:66][web:77]

    price = model.predict(features)[0]  # single prediction [web:69][web:77]

    st.write(
        {
            "carat": carat,
            "depth": depth,
            "table": table,
            "x": x,
            "y": y,
            "z": z,
            "cut": cut,
            "color": color,
            "clarity": clarity,
        }
    )

    st.subheader(f"Predicted Price: ${price:,.2f}")
