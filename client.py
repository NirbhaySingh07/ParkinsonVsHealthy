import librosa
import numpy as np
import requests
import sys

def analyze_voice_and_predict(audio_path):
    print(f"Loading and analyzing audio file: {audio_path}...")
    
    try:
        # 1. Load the raw audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # 2. Extract exactly 13 MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # 3. Calculate mean and std deviation across the time axis
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_stds = np.std(mfccs, axis=1)
        
        # 4. Map the numpy arrays to the strict Pydantic JSON schema
        payload = {}
        for i in range(13):
            payload[f"MFCC{i+1}_mean"] = float(mfcc_means[i])
            payload[f"MFCC{i+1}_std"] = float(mfcc_stds[i])
            
        print("Features extracted successfully. Sending to API...")
        
        # 5. Send the HTTP POST request to your local FastAPI server
        response = requests.post("http://localhost:8090/api/v1/predict", json=payload)
        
        # 6. Handle and display the result
        if response.status_code == 200:
            result = response.json()
            print("\n" + "="*30)
            print("      PREDICTION RESULT      ")
            print("="*30)
            print(f"Diagnosis:  {result['predicted_class']}")
            confidence = result['probabilities'][result['predicted_class']]
            print(f"Confidence: {confidence:.1%}")
            print(f"Latency:    {result['inference_latency_ms']} ms")
            print("="*30 + "\n")
        else:
            print(f"API Error: HTTP {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Failed to process audio: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_audio_file.wav>")
    else:
        analyze_voice_and_predict(sys.argv[1])