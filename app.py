import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go

MODEL_DIR = "models"
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier.joblib")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")
OPTIMAL_THRESHOLD = 0.2

def generate_recommendations(prob: float, user_input):
    suggestions = []
    
    if user_input['flight_hour'].iloc[0] > 10 and user_input['wants_in_flight_meals'].iloc[0] == 0:
        suggestions.append({
            "icon": "✈️",
            "title": "In-Flight Meals",
            "description": "The flight is in the daytime. Consider suggesting in-flight meals to enhance the customer experience.",
            "priority": "medium"
        })

    if user_input['flight_duration'].iloc[0] > 5 and user_input['flight_hour'].iloc[0] > 10:
        suggestions.append({
            "icon": "🍽️",
            "title": "Meal Service Upgrade",
            "description": "Journey is after 10 AM with long duration. In-flight meals might be a valuable add-on.",
            "priority": "high"
        })

    if 17 < user_input['flight_duration'].iloc[0] < 24:
        suggestions.append({
            "icon": "💺",
            "title": "Preferred Seating",
            "description": "Night flight detected. Suggest preferred seats to ensure comfortable sleep and travel.",
            "priority": "high"
        })

    if user_input['flight_duration'].iloc[0] > 5 and user_input['wants_extra_baggage'].iloc[0] == 0:
        suggestions.append({
            "icon": "🧳",
            "title": "Extra Baggage",
            "description": "For longer flights, extra baggage allowance prevents last-minute issues.",
            "priority": "medium"
        })

    return suggestions

def get_probability_recommendations(prob: float):
    if prob >= 0.75:
        return {
            "level": "Very High",
            "color": "#28a745",
            "icon": "🎉",
            "recommendations": [
                "🌟 Upgrade to Premium Class - Enjoy priority boarding and extra legroom for just 20% more!",
                "🛡️ Secure your trip with Travel Insurance - Complete protection for only $29",
                "✈️ Add Airport Lounge Access - Relax in luxury before your flight for $45"
            ]
        }
    elif 0.50 <= prob < 0.75:
        return {
            "level": "Moderate",
            "color": "#ffc107",
            "icon": "🙂",
            "recommendations": [
                "⚡ Limited Time: 15% OFF if you book in the next 2 hours!",
                "💳 Flexible Payment - Pay in 3 easy installments with 0% interest",
                "🔄 FREE Cancellation up to 24 hours before departure - Book with confidence!"
            ]
        }
    else:
        return {
            "level": "Low",
            "color": "#dc3545",
            "icon": "🤔",
            "recommendations": [
                "💰 Save 25% by flying 2 days earlier or later - Check alternative dates now!",
                "🎁 Join our Loyalty Program and get 500 bonus points + 10% off this booking",
                "📅 Book now, pay later - Reserve your seat for just $50 and pay the rest in 30 days"
            ]
        }

@st.cache_resource
def load_assets():
    if not os.path.exists(CLASSIFIER_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        st.error("Model files not found. Please run the training script first to create the 'models' directory and save the files.")
        st.stop()
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    classifier = joblib.load(CLASSIFIER_PATH)
    return preprocessor, classifier

# Page Configuration
st.set_page_config(
    page_title="Proactive Flight Booking System", 
    layout="wide", 
    initial_sidebar_state="collapsed",
    page_icon="✈️"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .prediction-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid #e1e5e9;
    }
    
    .recommendation-card {
        background: white;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 0.8rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border: 1px solid #e1e5e9;
    }
    
    .high-priority {
        border-left-color: #e74c3c !important;
        background: #fff5f5;
    }
    
    .medium-priority {
        border-left-color: #f39c12 !important;
        background: #fffaf0;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .probability-display {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        width: 100%;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
    }
    
    .recommendation-title {
        color: #333;
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .recommendation-desc {
        color: #666;
        font-size: 0.95rem;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>✈️ Proactive Flight Booking & Recommendation System </h1>
    <p>Predicting the booking completion probability and receive dynamic, actionable recommendations</p>
</div>
""", unsafe_allow_html=True)

preprocessor, classifier = load_assets()

# Input Section
st.markdown("## 📊 Customer Information")
st.markdown("Enter customer details to generate personalized recommendations")

with st.container():
    input_col1, input_col2, input_col3 = st.columns([1, 1, 1])
    
    with input_col1:
        st.markdown("### 👥 Trip Details")
        num_passengers = st.number_input("Number of Passengers", value=1, min_value=1, step=1)
        sales_channel = st.selectbox("Sales Channel", ["Internet", "Mobile"])
        trip_type = st.selectbox("Trip Type", ["RoundTrip", "OneWay", "CircleTrip"])
        purchase_lead = st.number_input("Purchase Lead (days)", value=10, min_value=0, step=1)
        
    
    with input_col2:
        st.markdown("### ✈️ Flight Information")
        length_of_stay = st.number_input("Length of Stay (days)", value=5, min_value=0, step=1)
        flight_hour = st.number_input("Flight Hour (24h format)", value=18, min_value=0, max_value=23, step=1)
        flight_duration = st.number_input("Flight Duration (hours)", value=6.5, min_value=0.0, step=0.1)
        flight_day = st.selectbox("Flight Day", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        
    
    with input_col3:
        st.markdown("### 🛍️ Preferences & Booking")
        route = st.selectbox("Route", options=['AKLDEL', 'AKLHGH', 'AKLHND', 'AKLICN', 'AKLKIX', 'AKLKTM', 'AKLKUL'])
        booking_origin = st.selectbox("Booking Origin", options=['New Zealand', 'India', 'China', 'South Korea', 'Japan', 'Malaysia'])
        st.markdown("### 🎯 Add-on Services")
        wants_extra_baggage = st.checkbox("Extra Baggage")
        wants_preferred_seat = st.checkbox("Preferred Seat")
        wants_in_flight_meals = st.checkbox("In-Flight Meals")

# Analyze Button
st.markdown("<br>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    analyze_clicked = st.button("🚀 Analyze & Get Recommendations", type="primary", use_container_width=True)

if analyze_clicked:
    user_input = pd.DataFrame({
        'num_passengers': [num_passengers],
        'sales_channel': [sales_channel],
        'trip_type': [trip_type],
        'purchase_lead': [purchase_lead],
        'length_of_stay': [length_of_stay],
        'flight_hour': [flight_hour],
        'flight_day': [flight_day],
        'route': [route],
        'booking_origin': [booking_origin],
        'wants_extra_baggage': [int(wants_extra_baggage)],
        'wants_preferred_seat': [int(wants_preferred_seat)],
        'wants_in_flight_meals': [int(wants_in_flight_meals)],
        'flight_duration': [flight_duration]
    })

    user_input_transformed = preprocessor.transform(user_input)
    proba = float(classifier.predict_proba(user_input_transformed)[0][1])
    
    st.markdown("---")
    st.markdown("## 🎯 Analysis Results")
    
    # Results Layout - Single column for probability display
    st.markdown("### 📊 Booking Probability")
    prob_data = get_probability_recommendations(proba)
    
    st.markdown(f"""
    <div class="insight-box">
        <h3>{prob_data['icon']} Probability Level: {prob_data['level']}</h3>
        <div class="probability-display">{proba:.1%}</div>
        <p>Based on the customer profile and historical data patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats in a row
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Passengers", num_passengers)
    with col_m2:
        st.metric("Flight Duration", f"{flight_duration}h")
    with col_m3:
        st.metric("Purchase Lead", f"{purchase_lead}d")
    
    # Recommendations Section
    st.markdown("## 💡 Strategic Recommendations")
    
    rec_col1, rec_col2 = st.columns([1, 1])
    
    with rec_col1:
        st.markdown("### 🎯 Probability-Based Actions")
        prob_data = get_probability_recommendations(proba)
        
        for i, rec in enumerate(prob_data['recommendations'], 1):
            st.markdown(f"""
            <div class="recommendation-card">
                <div class="recommendation-title">{i}. {rec}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with rec_col2:
        st.markdown("### 🛍️ Service Suggestions")
        suggestions = generate_recommendations(proba, user_input)
        
        if suggestions:
            for suggestion in suggestions:
                priority_class = f"{suggestion['priority']}-priority" if suggestion['priority'] in ['high', 'medium'] else ""
                st.markdown(f"""
                <div class="recommendation-card {priority_class}">
                    <div class="recommendation-title">{suggestion['icon']} {suggestion['title']}</div>
                    <div class="recommendation-desc">{suggestion['description']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="recommendation-card">
                <div class="recommendation-title">✅ Optimal Selection</div>
                <div class="recommendation-desc">Customer has already selected key options or their trip details don't suggest additional specific add-ons.</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Summary Section
    # st.markdown("## 📋 Action Summary")
    # summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    # with summary_col1:
    #     st.info(f"**Booking Likelihood:** {prob_data['level']}")
    
    # with summary_col2:
    #     st.info(f"**Key Focus:** {'Upselling' if proba >= 0.75 else 'Conversion' if proba >= 0.5 else 'Retention'}")
    
    # with summary_col3:
    #     suggestion_count = len(generate_recommendations(proba, user_input))
    #     st.info(f"**Service Suggestions:** {suggestion_count} recommendations")