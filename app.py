import streamlit as st
import pandas as pd
import joblib
import re
from textblob import TextBlob
import datetime 
@st.cache_resource
def load_ml_assets():
    try:
        model = joblib.load('models/random_forest_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        model_features = joblib.load('models/model_features.pkl')
        original_categorical_cols = joblib.load('models/original_categorical_cols.pkl')
        priority_map = joblib.load('models/priority_map.pkl')
        return model, scaler, model_features, original_categorical_cols, priority_map
    except FileNotFoundError as e:
        st.error(f"Error: Missing model asset file. Please ensure all .pkl files are in the 'models/' directory.")
        st.error(f"Details: {e}")
        st.stop()

model, scaler, model_features, original_categorical_cols, priority_map = load_ml_assets()

# Preprocessing Functions

def clean_text_pipeline(text):
    """Applies basic text cleaning steps from the notebook."""
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'\{.*?\}', '', text) 
    text = re.sub(r'\\+', '', text)      
    text = re.sub(r'[^a-z\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip() 
    return text

def preprocess_input_data(input_data_dict):
    """
    Applies the same preprocessing and feature engineering steps to new input data
    as done during model training in the notebook.
    """
    df_temp = pd.DataFrame([input_data_dict]) 

    # Text Cleaning and Text-based Feature Engineering
    df_temp['Ticket Description'] = df_temp['Ticket Description'].apply(clean_text_pipeline)
    df_temp['desc_word_count'] = df_temp['Ticket Description'].str.split().str.len()
    pattern = r'\burgent\b|\basap\b|\bimmediately\b|\bemergency\b|\bnow\b|\bcritical\b'
    df_temp['desc_has_urgent'] = df_temp['Ticket Description'].str.contains(pattern, case=False).astype(int)
    df_temp['desc_sentiment'] = df_temp['Ticket Description'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Date Handling - Extract year/month from 'Date of Purchase'
    df_temp['Date of Purchase'] = pd.to_datetime(df_temp['Date of Purchase'], errors='coerce')
    df_temp['purchase_year'] = df_temp['Date of Purchase'].dt.year
    df_temp['purchase_month'] = df_temp['Date of Purchase'].dt.month
    df_temp['purchase_year'] = df_temp['purchase_year'].fillna(df_temp['purchase_year'].mode()[0] if not df_temp['purchase_year'].empty else datetime.date.today().year).astype(int)
    df_temp['purchase_month'] = df_temp['purchase_month'].fillna(df_temp['purchase_month'].mode()[0] if not df_temp['purchase_month'].empty else datetime.date.today().month).astype(int)

    # Handle 'First Response Time'
    df_temp['First Response Time'] = pd.to_datetime(df_temp['First Response Time'], errors='coerce')
    df_temp['response_hour'] = df_temp['First Response Time'].dt.hour
    df_temp['response_dayofweek'] = df_temp['First Response Time'].dt.dayofweek
    df_temp['response_hour'] = df_temp['response_hour'].fillna(df_temp['response_hour'].mode()[0] if not df_temp['response_hour'].empty else 12).astype(int)
    df_temp['response_dayofweek'] = df_temp['response_dayofweek'].fillna(df_temp['response_dayofweek'].mode()[0] if not df_temp['response_dayofweek'].empty else 3).astype(int)

    columns_to_drop_pre_ohe = [
        'Ticket ID', 'Customer Name', 'Customer Email', 'Resolution', 'Time to Resolution', 'Customer Satisfaction Rating', # Dropped initially in notebook
        'Ticket Description', 'Date of Purchase', 'First Response Time', # Original columns replaced by engineered features
        'Product Purchased', 'Ticket Subject' # Dropped from X in notebook, not used as features
    ]
    df_processed_features = df_temp.drop(columns=[col for col in columns_to_drop_pre_ohe if col in df_temp.columns], errors='ignore')

    # One-Hot Encode the relevant original categorical columns
    df_ohe = pd.get_dummies(df_processed_features, columns=original_categorical_cols, drop_first=True)
    
    # Convert boolean columns to int
    bool_cols = df_ohe.select_dtypes(include='bool').columns
    df_ohe[bool_cols] = df_ohe[bool_cols].astype(int)

    final_features_df = df_ohe.reindex(columns=model_features, fill_value=0)

    # Scale numerical features
    numerical_cols_to_scale = scaler.feature_names_in_ 
    cols_to_scale_present = [col for col in numerical_cols_to_scale if col in final_features_df.columns]

    if cols_to_scale_present:
        final_features_df[cols_to_scale_present] = scaler.transform(final_features_df[cols_to_scale_present])
    else:
        st.warning("No numerical columns found to scale based on scaler's fitted features. Ensure feature names match.")

    return final_features_df

# --- Streamlit Interface ---

st.set_page_config(page_title="Ticket Priority Classifier", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #ffffff; /* Light background */
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4CAF50; /* Green submit button */
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 22px;
        border: none;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        transition: background-color 0.3s ease, transform 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div {
        border-radius: 8px;
        border: 1px solid #dcdcdc; /* Light gray border */
        padding: 10px;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    h1, h2, h3 {
        color: #2c3e50; /* Dark blue-grey for headers */
        text-align: center;
        margin-bottom: 25px;
    }
    .stMarkdown {
        text-align: center;
        font-size: 1.1em;
        color: #555;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Customer Support Ticket Priority Classifier")
st.markdown("Predict the priority level of a new customer support ticket to streamline operations.")

# --- Input Fields ---
with st.form("ticket_form"):
    st.subheader("Ticket Details")
    
    col1, col2 = st.columns(2)

    with col1:
        customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=30, help="Age of the customer.")
        customer_gender = st.selectbox("Customer Gender", ["Male", "Female", "Other"], help="Customer's self-identified gender.")
        
        ticket_type = st.selectbox("Ticket Type", [
            "Technical issue", "Billing inquiry", "Product inquiry", "Cancellation request", "Refund request"
        ], help="The nature of the customer's problem.")
        
        ticket_status = st.selectbox("Ticket Status", [
            "Open", "Pending Customer Response", "Closed"
        ], help="Current status of the ticket.")
        
        ticket_channel = st.selectbox("Ticket Channel", [
            "Email", "Phone", "Chat", "Social media"
        ], help="How the customer contacted support.")

    with col2:
        date_of_purchase = st.date_input("Date of Purchase", datetime.date(2023, 1, 1), help="Date the product was purchased.")
        first_response_time = st.date_input("First Response Date (Dummy)", datetime.date(2023, 6, 1), help="Use a dummy date for now, as exact time is not crucial for the hour/dayofweek extraction in this model version.")
        first_response_time_str = f"{first_response_time} 12:00:00" # Dummy time for conversion
        
        ticket_description = st.text_area("Ticket Description", 
                                          "My device is not turning on. It's urgent, I need help immediately!", 
                                          height=150, help="A detailed description of the issue. Look out for urgent keywords!")

    submitted = st.form_submit_button("Predict Priority")

# --- Prediction Logic ---
if submitted:
    input_data = {
        'Customer Age': customer_age,
        'Customer Gender': customer_gender,
        'Product Purchased': 'Dummy Product', 
        'Date of Purchase': str(date_of_purchase),
        'Ticket Type': ticket_type,
        'Ticket Subject': 'Dummy Subject',
        'Ticket Description': ticket_description,
        'Ticket Status': ticket_status,
        'Ticket Channel': ticket_channel,
        'First Response Time': first_response_time_str 
    }

    try:
        # Preprocess input and make prediction
        processed_input_df = preprocess_input_data(input_data)
        prediction_encoded = model.predict(processed_input_df)[0]
        predicted_priority = priority_map.get(prediction_encoded, "Unknown")

        st.success(f"**Predicted Ticket Priority: {predicted_priority}**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please check your input values and the integrity of the model assets in the 'models/' folder.")

st.markdown("---")