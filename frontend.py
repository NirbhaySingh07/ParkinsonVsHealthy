import streamlit as st
import librosa
import numpy as np
import requests

# Configure the page
st.set_page_config(page_title="Acoustic Classifier", page_icon="🎙️", layout="centered")

st.title("🎙️ Parkinson's Acoustic Classifier")
st.write("Upload a raw `.wav` audio sample to analyze dysarthric voice features and detect Parkinson's Disease (PD) vs. Healthy Control (HC).")

# 1. File Uploader UI
uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

if uploaded_file is not None:
    # Play the uploaded audio
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Analyze Audio", type="primary", use_container_width=True):
        with st.spinner("Extracting MFCC features and running inference..."):
            try:
                # 2. Extract Features using librosa
                # librosa expects a file path or file-like object; Streamlit provides a file-like object
                y, sr = librosa.load(uploaded_file, sr=None)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                
                mfcc_means = np.mean(mfccs, axis=1)
                mfcc_stds = np.std(mfccs, axis=1)
                
                # 3. Format the payload for your FastAPI backend
                payload = {}
                for i in range(13):
                    payload[f"MFCC{i+1}_mean"] = float(mfcc_means[i])
                    payload[f"MFCC{i+1}_std"] = float(mfcc_stds[i])
                
                # 4. Call the API
                api_url = "https://parkinsonvshealthy.onrender.com/api/v1/predict"
                response = requests.post(api_url, json=payload)
                
                # 5. Display the Results
                if response.status_code == 200:
                    result = response.json()
                    diagnosis = result['predicted_class']
                    confidence = result['probabilities'][diagnosis]
                    latency = result['inference_latency_ms']
                    
                    st.divider()
                    
                    # Highlight the diagnosis
                    if diagnosis == "PD":
                        st.error(f"### 🚨 Diagnosis: Parkinson's Disease (PD)")
                    else:
                        st.success(f"### ✅ Diagnosis: Healthy Control (HC)")
                    
                    # Show metrics in columns
                    col1, col2 = st.columns(2)
                    col1.metric("Confidence", f"{confidence:.1%}")
                    col2.metric("Inference Latency", f"{latency} ms")
                    
                    # Visual confidence bar
                    st.write("**Model Confidence:**")
                    st.progress(float(confidence))
                    
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")