import streamlit as st
import pickle
import pandas as pd

# Load Model yang sudah disimpan
model = pickle.load(open('Dataset/Model_final.pkl', 'rb'))

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
        "is_repeated_guest": is_repeated_guest
    }

    return pd.DataFrame([data])

# Ambil input dari user
input_df = user_input()

# Pastikan fitur input_df sesuai dengan model
expected_features = model.feature_names_in_  # Daftar fitur yang digunakan saat training

# Tambahkan fitur yang hilang dengan nilai default 0
for feature in expected_features:
    if feature not in input_df.columns:
        input_df[feature] = 0  

# Pastikan urutan kolom sesuai dengan model
input_df = input_df[expected_features]

# Tampilkan input data
st.subheader("Data Pemesanan yang Dimasukkan:")
st.write(input_df)

# Prediksi dengan model
if st.button("Prediksi Pembatalan"):
    prediction = model.predict(input_df)

    # Interpretasi hasil
    if prediction[0] == 1:
        st.error("\U0001F6A8 Booking kemungkinan besar akan DIBATALKAN!")
    else:
        st.success("✅ Booking kemungkinan besar akan DILANJUTKAN!")
