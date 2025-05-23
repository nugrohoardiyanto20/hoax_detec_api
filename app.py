import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import string
import os
import requests
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Set page config
st.set_page_config(page_title="Validin", page_icon="📰", layout="wide")

# Set custom NLTK data directory to a writable location
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Download NLTK data with error handling
try:
    nltk.download('punkt_tab', download_dir=nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir)
except Exception as e:
    st.error(f"Failed to download NLTK data: {str(e)}. Please ensure you have internet access and write permissions.")
    st.stop()

# Inisialisasi NLTK
stop_words = set(stopwords.words('indonesian'))

# Konfigurasi Grok AI
GROK_API_KEY = "xai-oANOG2INjZRhmPtTbDBwNNQvYWtrfZGr67msIs0jZWG9OOq9b99qeYXH88nEso37hSKiXREdhUnD1mVh"
GROK_API_URL = "https://api.x.ai/v1/chat/completions"

# Fungsi preprocessing teks
def clean(text):
    text = str(text).lower()
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    punct = set(string.punctuation)
    text = "".join([ch for ch in text if ch not in punct])
    return text

def tokenize(text):
    return word_tokenize(text)

def remove_stop_words(text):
    word_tokens_no_stopwords = [w for w in text if w not in stop_words]
    return word_tokens_no_stopwords

def preprocess(text):
    text = clean(text)
    text = tokenize(text)
    text = remove_stop_words(text)
    return text

# Cache model dan tokenizer
@st.cache_resource
def load_lstm_model():
    return load_model('hoax_lstm_model.h5')

@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as handle:
        return pickle.load(handle)

# Fungsi untuk mendapatkan rekomendasi dari Grok AI
def get_grok_recommendations(news_text, prediction_result, confidence):
    """
    Mendapatkan rekomendasi dari Grok AI berdasarkan hasil prediksi
    """
    try:
        headers = {
            "Authorization": f"Bearer {GROK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Tentukan status berita
        status = "HOAX" if prediction_result == 1 else "VALID"
        
        # Buat prompt untuk Grok AI
        prompt = f"""
        Sebagai AI assistant yang ahli dalam analisis berita dan media literacy, berikan rekomendasi dan saran yang berguna untuk pengguna.

        Konteks:
        - Teks berita telah dianalisis dan dikategorikan sebagai: {status}
        - Tingkat kepercayaan: {confidence:.2f}%
        - Teks berita: "{news_text[:200]}..."

        Berikan rekomendasi dalam format berikut:
        1. **Analisis Singkat**: Jelaskan mengapa berita ini dikategorikan sebagai {status}
        2. **Saran Verifikasi**: 3-4 langkah konkret untuk memverifikasi kebenaran berita
        3. **Tips Media Literacy**: 2-3 tips praktis untuk mengidentifikasi berita hoax di masa depan
        4. **Tindakan yang Disarankan**: Apa yang sebaiknya dilakukan pengguna selanjutnya

        Berikan jawaban dalam bahasa Indonesia yang mudah dipahami dan praktis.
        """
        
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": "grok-beta",
            "stream": False,
            "temperature": 0.7
        }
        
        response = requests.post(GROK_API_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"Error: Tidak dapat terhubung ke Grok AI (Status: {response.status_code})"
            
    except requests.exceptions.Timeout:
        return "Error: Timeout - Grok AI tidak merespons dalam waktu yang ditentukan"
    except requests.exceptions.RequestException as e:
        return f"Error: Masalah koneksi ke Grok AI - {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

# Muat model dan tokenizer
try:
    model = load_lstm_model()
    tokenizer = load_tokenizer()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Parameter tokenisasi
max_features = 5000
max_len = 300

# Custom CSS untuk tampilan modern
st.markdown("""
    <style>
    /* Reset default Streamlit styles */
    .stApp {
        background-color: #F3F4F6;
        font-family: 'Inter', sans-serif;
    }
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    /* Main container */
    .main {
        background-color: #F3F4F6;
        min-height: 100vh;
        padding: 2rem;
    }
    /* Header */
    .title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    /* Card untuk input */
    .card {
        background-color: #FFFFFF;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0 auto;
        max-width: 800px;
    }
    /* Text Area */
    div[data-testid="stTextArea"] textarea {
        border: 1px solid #D1D5DB !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        background-color: #F9FAFB !important;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #1E3A8A !important;
        box-shadow: 0 0 0 3px rgba(30, 58, 138, 0.1) !important;
    }
    /* Button */
    div[data-testid="stButton"] button {
        background-color: #1E3A8A !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        transition: background-color 0.3s ease !important;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #1C2F6B !important;
    }
    /* Result Box */
    .result-box {
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1.5rem;
        font-size: 1.1rem;
        font-weight: 500;
    }
    .success {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #10B981;
    }
    .error {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #EF4444;
    }
    /* Recommendation Box */
    .recommendation-box {
        background-color: #F0F9FF;
        border: 1px solid #0EA5E9;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }
    .recommendation-title {
        color: #0369A1;
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .recommendation-content {
        color: #1E40AF;
        line-height: 1.6;
        white-space: pre-wrap;
    }
    /* Footer */
    .footer {
        text-align: center;
        color: #6B7280;
        margin-top: 3rem;
        font-size: 0.9rem;
    }
    /* Loading animation */
    .loading-text {
        color: #0369A1;
        font-style: italic;
        text-align: center;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# UI Streamlit
st.markdown('<div class="main">', unsafe_allow_html=True)

# Header
st.markdown('<h1 class="title">VALIDIN</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Deteksi berita hoax dengan cepat dan akurat menggunakan AI canggih + Rekomendasi dari Grok AI.</p>', unsafe_allow_html=True)

# Konten utama dalam card
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Input teks
    news_text = st.text_area("Masukkan Teks Berita", placeholder="Tempel teks berita di sini...", height=200)

    # Tombol prediksi
    if st.button("🔍 Periksa Sekarang", type="primary"):
        if news_text.strip() == "":
            st.markdown('<div class="result-box error">⚠️ Mohon masukkan teks berita!</div>', unsafe_allow_html=True)
        else:
            with st.spinner("Menganalisis berita..."):
                # Preprocessing teks
                processed_text = preprocess(news_text)
                text_seq = tokenizer.texts_to_sequences([" ".join(processed_text)])
                if not text_seq[0]:
                    st.markdown('<div class="result-box error">⚠️ Teks tidak dapat diproses. Pastikan teks relevan.</div>', unsafe_allow_html=True)
                    st.stop()
                text_padded = pad_sequences(sequences=text_seq, maxlen=max_len, padding='pre')
                
                # Prediksi
                prediction = model.predict(text_padded)
                threshold = 0.6
                hoax_prob = prediction[0][1]
                pred_class = 1 if hoax_prob > threshold else 0
                pred_prob = hoax_prob * 100 if pred_class == 1 else (1 - hoax_prob) * 100

                # Tampilkan hasil
                if pred_class == 1:
                    st.markdown(f'<div class="result-box error">🚨 <b>Peringatan</b>: Berita ini kemungkinan <b>HOAX</b> (Kepercayaan: {pred_prob:.2f}%)</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-box success">✅ <b>Hasil</b>: Berita ini kemungkinan <b>VALID</b> (Kepercayaan: {pred_prob:.2f}%)</div>', unsafe_allow_html=True)

            # Mendapatkan rekomendasi dari Grok AI
            with st.spinner("Mendapatkan rekomendasi dari Grok AI..."):
                st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                st.markdown('<div class="recommendation-title">🤖 Rekomendasi dari Grok AI</div>', unsafe_allow_html=True)
                
                recommendations = get_grok_recommendations(news_text, pred_class, pred_prob)
                
                if recommendations.startswith("Error:"):
                    st.markdown(f'<div class="recommendation-content" style="color: #DC2626;">❌ {recommendations}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="recommendation-content">💡 <b>Tips Manual:</b><br/>' +
                              '• Verifikasi dengan sumber berita terpercaya<br/>' +
                              '• Cek fakta di situs fact-checking<br/>' +
                              '• Jangan langsung share tanpa verifikasi<br/>' +
                              '• Perhatikan tanda-tanda berita hoax seperti judul sensasional</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="recommendation-content">{recommendations}</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Informasi tambahan
with st.expander("ℹ️ Tentang Aplikasi"):
    st.write("""
    **Validin** adalah aplikasi deteksi berita hoax yang menggunakan:
    - **Model LSTM** untuk klasifikasi teks berita
    - **Grok AI** untuk memberikan rekomendasi dan saran verifikasi
    - **Analisis NLP** untuk preprocessing teks bahasa Indonesia
    
    **Cara Kerja:**
    1. Masukkan teks berita yang ingin diverifikasi
    2. Sistem akan menganalisis dan memberikan prediksi
    3. Grok AI akan memberikan rekomendasi berdasarkan hasil analisis
    """)

# Footer
st.markdown('<p class="footer">© 2025 Validin - Powered by AI & Grok AI.</p>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
