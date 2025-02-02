import streamlit as st
import pandas as pd
import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import skew, kurtosis
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

# Load your dataset and model
#@st.cache_data 
def load_data_and_model():
    # Replace with your actual dataset
    clustering_model = joblib.load("clustering_model.pkl")
    songs_df = pd.read_csv("recommendation_features1.csv")  # Your dataset with song features
    return songs_df, clustering_model

songs_df, clustering_model = load_data_and_model()

# Function to extract spectral features from an audio file
def extract_features(audio_file, sr=22050):
    try:
        # Load the audio file
        y, sr = librosa.load(audio_file, sr=sr)
        
        # Spectral Features
        features = {}
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_var'] = np.var(mfccs, axis=1)

        # Feature Trajectories
        mfcc_delta = librosa.feature.delta(mfccs)
        features['mfcc_delta_mean'] = np.mean(mfcc_delta, axis=1)
        features['mfcc_delta_var'] = np.var(mfcc_delta, axis=1)
        
        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_var'] = np.var(spectral_centroid)
        
        # Spectral Roll-off
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_var'] = np.var(spectral_rolloff)
        
        # Spectral Flux
        spectral_flux = librosa.onset.onset_strength(y=y, sr=sr)
        features['spectral_flux_mean'] = np.mean(spectral_flux)
        features['spectral_flux_var'] = np.var(spectral_flux)
        
        # Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features['spectral_contrast_mean'] = np.mean(spectral_contrast, axis=1)
        features['spectral_contrast_var'] = np.var(spectral_contrast, axis=1)

        envelope = np.abs(y)
        features['envelope_mean'] = np.mean(envelope)
        features['envelope_var'] = np.var(envelope)

        # Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_var'] = np.var(spectral_bandwidth)

        # Time-Domain Features
        features['rms_mean'] = np.mean(librosa.feature.rms(y=y))
        features['rms_var'] = np.var(librosa.feature.rms(y=y))
        features['energy'] = np.sum(y ** 2) / len(y)  # Signal energy
        features['amplitude_mean'] = np.mean(np.abs(y))
        features['amplitude_var'] = np.var(np.abs(y))

        # Temporal Evolution
        dynamic_range = np.max(y) - np.min(y)
        features['dynamic_range'] = dynamic_range

        # Rhythm Features
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        beat_strength = librosa.onset.onset_strength(y=y, sr=sr)
        features['beat_strength_mean'] = np.mean(beat_strength)
        features['beat_strength_var'] = np.var(beat_strength)

        # Rhythmic Regularity
        if len(beat_frames) > 1:
            # Inter-Beat Interval (IBI) Variability
            ibi = np.diff(beat_frames) / sr  # Convert frame difference to seconds
            features['ibi_var'] = np.var(ibi)
            features['ibi_mean'] = np.mean(ibi)

        # Harmonic Features
        # Key and Scale Estimation
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        features['chroma_mean'] = np.mean(chroma, axis=1)
        features['chroma_var'] = np.var(chroma, axis=1)
        
        # Tonnetz
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        features['tonnetz_mean'] = np.mean(tonnetz, axis=1)
        features['tonnetz_var'] = np.var(tonnetz, axis=1)

        # Harmonic-to-Percussive Ratio
        harmonic, percussive = librosa.effects.hpss(y)
        hpr = np.mean(harmonic) / (np.mean(percussive) + 1e-6)
        features['hpr'] = hpr

        # Genre-Specific Features
        # Harmonic and Percussive Energy
        features['harmonic_energy'] = np.sum(harmonic ** 2)
        features['percussive_energy'] = np.sum(percussive ** 2)

        # Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_var'] = np.var(zcr)

        # Onset Autocorrelation (Rhythmic Regularity)
        onset_acf = librosa.autocorrelate(librosa.onset.onset_strength(y=y, sr=sr), max_size=len(y)//2)
        features['onset_acf_mean'] = np.mean(onset_acf)
        features['onset_acf_var'] = np.var(onset_acf)

        # Statistics (Skewness and Kurtosis)
        features['skewness'] = skew(y)
        features['kurtosis'] = kurtosis(y)
        
        # Combine features into a single vector
        feature_vector = np.hstack(list(features.values()))
        return feature_vector
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None



# Add clustering logic to the recommendation function
def recommend_songs(features, songs_df, clustering_model, num_recommendations=5):
    # Extract the feature columns from the dataset
    feature_cols = [col for col in songs_df.columns if col.startswith("feature_")]

    # Normalize the input song features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform([features])

    # Predict the cluster of the input song
    predicted_cluster = clustering_model.predict(features_scaled)[0]

    # Filter dataset to include only songs from the same cluster
    cluster_songs = songs_df[songs_df['cluster'] == predicted_cluster]

    # Compute similarity (e.g., cosine similarity) within the cluster
    cluster_features = cluster_songs[feature_cols].values
    distances = euclidean_distances([features], cluster_features).flatten()

    similarities = 1 / (1 + distances)
    cluster_songs['similarity'] = similarities

    # Sort by similarity and return top recommendations
    recommendations = cluster_songs.sort_values(by='similarity', ascending=False)
    return recommendations[['file_name', 'similarity', 'audio_path']].head(num_recommendations)


# Streamlit UI
st.title("Song Recommendation App 🎵")
st.write("Upload a song to get 5 similar song recommendations!")

# File uploader
uploaded_file = st.file_uploader("Upload a song file (MP3, WAV, etc.)", type=["mp3", "wav"])

if uploaded_file:
    # Display an audio player for the uploaded file
    st.subheader("Uploaded Song Preview:")
    st.audio(uploaded_file, format="audio/mp3")  # Play the uploaded song

    # Extract features from the uploaded file
    st.write("Processing uploaded song...")
    song_features = extract_features(uploaded_file)

    if song_features is not None:
        st.write("Finding similar songs...")

        # Recommend similar songs (with clustering logic)
        recommendations = recommend_songs(song_features, songs_df, clustering_model)

        st.write("Here are 5 similar songs:")

        # Display recommendations with audio playback
        for _, row in recommendations.iterrows():
            st.subheader(row['file_name'])
            st.write(f"Similarity Score: {row['similarity']:.2f}")
            st.audio(row['audio_path'], format="audio/mp3")
    else:
        st.error("Failed to extract features from the uploaded file.")

