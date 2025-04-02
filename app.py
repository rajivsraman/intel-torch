import streamlit as st
import pandas as pd
import os
from PIL import Image
import time
import random
import torch
from huggingface_hub import hf_hub_download
from trainmodel import create_model

def load_predictions():
    # Read the CSV file with predictions
    return pd.read_csv('predictions.csv')

def create_slideshow():
    st.title("Random Scene Prediction Demo")

    # Load predictions
    predictions_df = load_predictions()
    
    # Create a placeholder for the image and label
    image_placeholder = st.empty()
    label_placeholder = st.empty()
    
    # Add a start/stop button
    if 'running' not in st.session_state:
        st.session_state.running = False
    
    if st.button('Start/Stop Slideshow'):
        st.session_state.running = not st.session_state.running
    
    while st.session_state.running:
        # Get a random row from the dataframe
        random_row = predictions_df.iloc[random.randint(0, len(predictions_df) - 1)]
        
        filename = random_row['Filename']
        predicted_class = random_row['Predicted Class']
        
        # Load and display image
        image_path = os.path.join('data/pred', filename)
        try:
            image = Image.open(image_path)
            image_placeholder.image(image, caption=filename)
            label_placeholder.markdown(f"### Predicted Class: {predicted_class}")
            
            # Wait for 3 seconds
            time.sleep(3)
        except Exception as e:
            st.error(f"Error loading image {filename}: {str(e)}")

def main():
    create_slideshow()

if __name__ == "__main__":
    main()
