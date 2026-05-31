import streamlit as st
import pandas as pd
import joblib
from autogluon.tabular import TabularPredictor
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

@st.cache_resource
def load_model_and_scaler():
    model_path = os.path.join(BASE_DIR, "agModel")
    predictor = TabularPredictor.load(model_path, require_py_version_match=False)
    scaler = joblib.load(os.path.join(model_path, "scaler.pkl"))
    return predictor, scaler

predictor, scaler = load_model_and_scaler()

@st.cache_data
def load_reference_data():
    data_path = os.path.join(BASE_DIR, "data", "cleaned.csv")
    return pd.read_csv(data_path)

ref_df = load_reference_data()

st.title("Car Price Predictor")
st.write("Enter car specifics to get an estimated price.")

brand = st.selectbox("Brand", sorted(ref_df["Brand"].unique()))
model = st.selectbox("Model", sorted(ref_df["Model"].unique()))
production_year = st.slider(
    "Production Year",
    min_value=int(ref_df["Year"].min()),
    max_value=int(ref_df["Year"].max()),
    value=int(ref_df["Year"].median())
)
engine_size = st.number_input(
    "Engine Size (L)",
    min_value=float(ref_df["Engine_Size"].min()),
    max_value=float(ref_df["Engine_Size"].max()),
    value=float(ref_df["Engine_Size"].median()),
    step=0.1
)
fuel_type = st.selectbox("Fuel Type", sorted(ref_df["Fuel_Type"].unique()))
transmission = st.selectbox("Transmission", sorted(ref_df["Transmission"].unique()))
mileage = st.number_input(
    "Mileage (km)",
    min_value=int(ref_df["Mileage"].min()),
    max_value=int(ref_df["Mileage"].max()),
    value=int(ref_df["Mileage"].median())
)
doors = st.selectbox("Doors", sorted(ref_df["Doors"].unique()))
owner_count = st.selectbox("Owner Count", sorted(ref_df["Owner_Count"].unique()))

if st.button("Predict Price"):
    input_df = pd.DataFrame([{
        "Brand": brand,
        "Model": model,
        "Year": production_year,
        "Engine_Size": engine_size,
        "Fuel_Type": fuel_type,
        "Transmission": transmission,
        "Mileage": mileage,
        "Doors": doors,
        "Owner_Count": owner_count
    }])

    num_cols = ["Year", "Engine_Size", "Mileage", "Doors", "Owner_Count"]
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    prediction = predictor.predict(input_df)[0]

    st.success(f"Estimated Price: **${prediction:,.2f}**")
