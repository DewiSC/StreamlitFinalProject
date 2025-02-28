import streamlit as st
import pickle
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load Model yang sudah disimpan
# model = pickle.load(open('Dataset/Model_final.pkl', 'rb'))

base_path = os.path.dirname(os.path.abspath(__file__))  # Path ke direktori script
model_path = os.path.join(base_path, "Dataset", "Model_final.pkl")

model = pickle.load(open(model_path, 'rb'))

# Load Encoder yang sudah dilatih
encoder = pickle.load(open('Dataset/encoder.pkl', 'rb'))  # Encoder harus disimpan saat training

# Judul Aplikasi
st.title("Hotel Booking Cancellation Prediction")

# Form input untuk memasukkan fitur prediksi
st.sidebar.header("Masukkan Data Pemesanan:")

def user_input():
    lead_time = st.sidebar.slider("Lead Time (Hari)", 0, 365, 100)
    stays_in_week_nights = st.sidebar.slider("Stays in Week Nights", 0, 14, 2)
    adults = st.sidebar.slider("Adults", 1, 5, 2)
    previous_cancellations = st.sidebar.slider("Previous Cancellations", 0, 10, 0)
    previous_bookings_not_canceled = st.sidebar.slider("Previous Bookings Not Canceled", 0, 10, 0)
    booking_changes = st.sidebar.slider("Booking Changes", 0, 10, 0)
    required_car_parking_spaces = st.sidebar.slider("Required Car Parking Spaces", 0, 5, 0)
    total_special_requests = st.sidebar.slider("Total Special Requests", 0, 5, 1)
    length_of_stay = st.sidebar.slider("Length of Stay", 1, 30, 3)
    is_repeated_guest = st.sidebar.radio("Is Repeated Guest?", [0, 1])  # Binary input

    # Input kategorikal
    meal = st.sidebar.selectbox("Meal", ['BB', 'FB', 'HB', 'SC', 'Undefined'])
    market_segment = st.sidebar.selectbox("Market Segment", ['Online', 'Offline', 'Direct', 'Corporate'])
    distribution_channel = st.sidebar.selectbox("Distribution Channel", ['TA/TO', 'Direct', 'Corporate', 'GDS'])
    reserved_room_type = st.sidebar.selectbox("Reserved Room Type", ['Standard', 'Superior', 'Deluxe', 'Suite'])
    assigned_room_type = st.sidebar.selectbox("Assigned Room Type", ['Standard', 'Superior', 'Deluxe', 'Suite'])
    deposit_type = st.sidebar.selectbox("Deposit Type", ['No Deposit', 'Non Refund', 'Refundable'])
    customer_type = st.sidebar.selectbox("Customer Type", ['Transient', 'Contract', 'Group'])
    stay_category = st.sidebar.selectbox("Stay Category", ['Short', 'Medium', 'Long'])

    # Data dalam bentuk dictionary
    data = {
        "lead_time": lead_time,
        "stays_in_week_nights": stays_in_week_nights,
        "adults": adults,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,
        "booking_changes": booking_changes,
        "required_car_parking_spaces": required_car_parking_spaces,
        "total_of_special_requests": total_special_requests,
        "length_of_stay": length_of_stay,
        "is_repeated_guest": is_repeated_guest,
        "meal": meal,
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "reserved_room_type": reserved_room_type,
        "assigned_room_type": assigned_room_type,
        "deposit_type": deposit_type,
        "customer_type": customer_type,
        "stay_category": stay_category
    }

    return pd.DataFrame([data])

# Ambil input dari user
input_df = user_input()

# Tampilkan input data sebelum encoding
st.subheader("Data Pemesanan yang Dimasukkan (Sebelum Encoding):")
st.write(input_df)

# Pastikan fitur input_df sesuai dengan model
expected_features = model.feature_names_in_  # Daftar fitur yang digunakan saat training

# Terapkan encoder ke input user agar sesuai dengan format model
input_encoded = encoder.transform(input_df)

# Pastikan kolom sesuai dengan model
for feature in expected_features:
    if feature not in input_encoded.columns:
        input_encoded[feature] = 0  # Tambahkan fitur yang hilang dengan default 0

# Urutkan kolom sesuai model
input_encoded = input_encoded[expected_features]

# Tampilkan input data setelah encoding
st.subheader("Data Setelah Encoding:")
st.write(input_encoded)

# Prediksi dengan model
if st.button("Prediksi Pembatalan"):
    prediction = model.predict(input_encoded)

    # Interpretasi hasil
    if prediction[0] == 1:
        st.error("\U0001F6A8 Booking kemungkinan besar akan DIBATALKAN!")
    else:
        st.success("✅ Booking kemungkinan besar akan DILANJUTKAN!")
