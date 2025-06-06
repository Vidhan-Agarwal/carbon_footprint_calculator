import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and scaler (assumes files are in same directory)
model = joblib.load("model.pkl")
_, _, _, _, scaler = joblib.load("processed_data.pkl")

# Vehicle options and their order must match training data columns exactly!
vehicle_options = ["Bike", "Car", "EV", "Public Transport"]

# Vehicle emissions map for suggestions and calculations
vehicle_emissions_map = {
    'Car': 0.21,
    'Bike': 0.05,
    'Public Transport': 0.09,
    'EV': 0.03,
    'None': 0.00
}

# Page setup
st.set_page_config(page_title="🌍 Carbon Footprint Calculator", layout="centered")
st.image("https://cdn.pixabay.com/photo/2017/01/20/00/30/globe-1990277_1280.jpg", use_container_width=True)
st.title("🌱 AI-Powered Carbon Footprint Calculator")
st.caption("_Estimate your daily carbon footprint and get personalized lifestyle tips_ ✅")

# Sidebar with about info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This tool estimates your daily CO₂ footprint using a trained ML model.")
    st.metric(label="R² Score", value="0.83")
    st.metric(label="MAE", value="0.64 kg CO₂")
    st.write("Built with 💚 using Streamlit & Scikit-learn")

# User inputs
st.header("📥 Your Lifestyle Inputs")
km = st.slider("🚗 Daily distance travelled (in km)", 0, 100, 20)
vehicle = st.selectbox("🚙 Primary mode of transport", vehicle_options + ["None"])
meat = st.slider("🍗 Daily meat consumption (in grams)", 0, 500, 200)
electricity = st.slider("💡 Monthly electricity usage (in kWh)", 50, 1000, 300)
shopping = st.slider("🛍️ Shopping frequency (0 = rarely, 10 = often)", 0, 10, 5)

# Prepare inputs
if vehicle == "None":
    vehicle_encoded = np.array([[0, 0, 0, 0]])
else:
    vehicle_encoded = np.array([[1 if vehicle == v else 0 for v in vehicle_options]])

numeric_input = np.array([[km, meat, electricity, shopping]])
numeric_scaled = scaler.transform(numeric_input)
user_input_final = np.concatenate([numeric_scaled, vehicle_encoded], axis=1)

# Predict button
if st.button("📊 Predict My Footprint"):
    prediction = model.predict(user_input_final)[0]

    vehicle_factor = vehicle_emissions_map.get(vehicle, 0)
    travel_footprint = km * vehicle_factor
    meat_footprint = meat * 0.005
    electricity_footprint = electricity * 0.01
    shopping_footprint = shopping * 0.2

    contributions = {
        'Travel (distance & vehicle)': travel_footprint,
        'Meat consumption': meat_footprint,
        'Electricity usage': electricity_footprint,
        'Shopping frequency': shopping_footprint
    }
    sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)

    # Display result as metric
    st.header("📈 Your Carbon Footprint")
    st.metric(label="🌿 Daily Emissions", value=f"{prediction:.2f} kg CO₂")

    # Progress bar
    st.progress(min(prediction / 12, 1.0))

    # Categorized feedback
    if prediction < 5:
        st.success("✅ Low footprint! Great job keeping your emissions down.")
        st.balloons()
    elif 5 <= prediction < 9:
        st.info("🟡 Moderate footprint. Some lifestyle changes can help reduce your impact.")
    elif 9 <= prediction < 11:
        st.warning("⚠️ High footprint! Consider adjustments.")
    else:
        st.error("🚨 Very high footprint! Significant lifestyle changes are recommended.")

    # Top contributor
    if prediction >= 5:
        top_factor, top_value = sorted_contrib[0]
        st.subheader(f"🔍 Top Contributor: {top_factor} ({top_value:.2f} kg CO₂)")

        if top_factor == 'Travel (distance & vehicle)':
            if vehicle == 'Car' and km > 20:
                st.write("- 🚗 Consider reducing car travel or carpooling.")
                st.write("- 🚲 Try biking or using public transport more.")
                st.write("- ⚡ Switch to an EV if possible.")
            elif vehicle == 'Public Transport' and km > 20:
                st.write("- 🚌 Good use of public transport! Reduce travel distance if possible.")
            elif vehicle == 'EV' and km > 20:
                st.write("- 🔌 EV is efficient. Reducing distance helps more.")
        elif top_factor == 'Meat consumption':
            st.write("- 🥗 Reduce meat intake and eat more plant-based meals.")
        elif top_factor == 'Electricity usage':
            st.write("- 💡 Turn off devices when not in use.")
            st.write("- 🔋 Use energy-efficient appliances.")
        elif top_factor == 'Shopping frequency':
            st.write("- 🛍️ Limit unnecessary purchases.")
            st.write("- ♻️ Reuse, recycle, repair where possible.")

    # Pie chart of contributions
    st.subheader("📊 Contribution Breakdown")
    fig, ax = plt.subplots()
    labels, values = zip(*contributions.items())
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    # General tips
    with st.expander("💡 General Tips"):
        st.write("""
        - Use public transport, bike, or switch to EV.
        - Reduce meat consumption.
        - Use energy-efficient appliances.
        - Shop consciously and avoid overconsumption.
        """)