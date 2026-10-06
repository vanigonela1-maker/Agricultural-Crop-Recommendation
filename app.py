import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Page settings
st.set_page_config(
    page_title="Agricultural Crop Recommendation",
    page_icon="🌱",
    layout="wide"
)

# Website design
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
}

div.stButton > button {
    background-color: #2e7d32;
    color: white;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
}
.stNumberInput label {
    color: #1b5e20 !important;
    font-weight: bold;
    font-size: 16px;
}

div.stButton > button:hover {
    background-color: #1b5e20;
    color: white;
}
.stNumberInput label {
    color: #1b5e20 !important;
    font-weight: bold;
    font-size: 16px;
}
div[data-testid="stNumberInput"] input {
    background-color: #ffffff !important;
    color: #1b1b1b !important;
    border: 2px solid #a5d6a7 !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# Title

st.markdown(
    "<h1 style='text-align:center;color:#1b5e20;font-size:42px;font-weight:bold;'>🌱 Agricultural Crop Recommendation System</h1>",
    unsafe_allow_html=True
)
   
# Description
st.markdown(
    """
        <div style="
        text-align:center;
        font-size:18px;
        margin-bottom:25px;
        color:#4e342e;
    ">
        🌾 <b>Smart Farming with Machine Learning</b><br>
        Get a suitable crop recommendation based on soil nutrients
        and environmental conditions.
    </div>
    """,
    unsafe_allow_html=True
)

# Load dataset
data = pd.read_csv("data/Crop_recommendation.csv")

# Prepare data
X = data.drop("label", axis=1)
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.markdown(
    f"""
    <div style="
        background-color:#ffffff;
        padding:15px;
        border-radius:15px;
        text-align:center;
        border:2px solid #81c784;
        margin:20px 0;
    ">
        <h3 style="color:#1b5e20;">🎯 Model Accuracy</h3>
        <h2 style="color:#2e7d32;">{accuracy * 100:.2f}%</h2>
    </div>
    """,
    unsafe_allow_html=True
)


# Input heading
st.divider()

st.markdown(
    "<h2 style='text-align:center;color:#2e7d32;'>🌾 Soil & Weather Information</h2>",
    unsafe_allow_html=True
)

# Input columns

# Input columns
st.info(
    "💡 Enter the soil and weather values below\n\n"
    "These values help the system recommend the most suitable crop."
)
col1, col2 = st.columns(2)

with col1:
    N = st.number_input(
        "Nitrogen (N)",
        min_value=0.0,
        value=50.0
    )

    P = st.number_input(
        "Phosphorus (P)",
        min_value=0.0,
        value=50.0
    )

    K = st.number_input(
        "Potassium (K)",
        min_value=0.0,
        value=50.0
    )

    ph = st.number_input(
        "Soil pH",
        min_value=0.0,
        max_value=14.0,
        value=6.5
    )

with col2:
    temperature = st.number_input(
        "Temperature (°C)",
        value=25.0
    )

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        value=100.0
    )

# Recommendation button
if st.button(
    "🌾 Recommend Crop",
    use_container_width=True,
    type="primary"
):

    # Create input data
    input_data = pd.DataFrame({
        "N": [N],
        "P": [P],
        "K": [K],
        "temperature": [temperature],
        "humidity": [humidity],
        "ph": [ph],
        "rainfall": [rainfall]
    })

    # Predict crop
    crop = model.predict(input_data)[0]

    # Display recommendation
    st.markdown(
        f"""
        <div style="
            background-color:#dcedc8;
            padding:30px;
            border-radius:20px;
            text-align:center;
            margin-top:30px;
            border:3px solid #4caf50;
            box-shadow:0 4px 12px rgba(0,0,0,0.15);
        ">
            <h2>🌱 Recommended Crop</h2>
            <h1 style="color:#2e7d32;">{crop.upper()}</h1>
            <p>Based on the soil and weather conditions you entered.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Crop information
    crop_info = {
        "rice": "Rice requires plenty of water and warm temperatures.",
        "maize": "Maize grows well in fertile soil with moderate rainfall.",
        "chickpea": "Chickpea prefers well-drained soil and moderate temperatures.",
        "kidneybeans": "Kidney beans grow well in moderate temperatures and good soil.",
        "pigeonpeas": "Pigeon pea is suitable for warm climates and moderate rainfall.",
        "mothbeans": "Moth bean is drought tolerant and suitable for dry regions.",
        "mungbean": "Mung bean grows well in warm weather and well-drained soil.",
        "blackgram": "Black gram prefers warm temperatures and moderate rainfall.",
        "lentil": "Lentil grows well in cool and relatively dry conditions.",
        "pomegranate": "Pomegranate prefers dry climates and well-drained soil.",
        "banana": "Banana requires warm temperatures, high humidity and good water availability.",
        "mango": "Mango grows best in warm climates with suitable rainfall.",
        "grapes": "Grapes prefer warm weather and well-drained soil.",
        "watermelon": "Watermelon requires warm temperatures and adequate water.",
        "muskmelon": "Muskmelon grows well in warm and relatively dry conditions.",
        "apple": "Apple requires cooler temperatures and suitable soil conditions.",
        "orange": "Orange grows well in warm climates with adequate water.",
        "papaya": "Papaya prefers warm temperatures and well-drained soil.",
        "coconut": "Coconut requires warm temperatures, high humidity and good rainfall.",
        "cotton": "Cotton grows well in warm temperatures and moderate rainfall.",
        "jute": "Jute requires warm and humid conditions with sufficient rainfall.",
        "coffee": "Coffee prefers warm, humid conditions and well-drained soil."
    }

    information = crop_info.get(
        crop.lower(),
        "This crop is suitable for the conditions provided."
    )

    st.info(
        f"🌿 **About {crop.title()}:** {information}"
    )

# Footer
st.markdown(
    """
    <div style="
        text-align:center;
        margin-top:50px;
        padding:20px;
        border-top:1px solid #c8e6c9;
        color:#666;
    ">
        🌱 <b>Agricultural Crop Recommendation System</b><br>
        Machine Learning Based Smart Farming Project<br>
        <small>Developed for Academic Project</small>
    </div>
    """,
    unsafe_allow_html=True
)