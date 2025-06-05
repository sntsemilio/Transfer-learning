import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# App configuration
st.set_page_config(
    page_title="Bean Leaf Disease Classifier",
    page_icon="🌱",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load pre-trained models"""
    try:
        # Primary model (replace with your best performing model)
        primary_model = load_model('models/best_vgg16_transfer_model_finetuned.h5')
        
        # Backup model
        backup_model = load_model('models/best_baseline_cnn_model.h5')
        
        return primary_model, backup_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def main():
    st.title("Bean Leaf Disease Classification System")
    st.subheader("Automated Detection of Angular Leaf Spot, Bean Rust, and Healthy Leaves")
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Select Model",
        ["VGG16 Transfer Learning", "Baseline CNN"]
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Bean Leaf Image", 
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file is not None:
        # Load models
        primary_model, backup_model = load_models()
        
        if primary_model is not None:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Make prediction
            if st.button("Classify Disease"):
                with st.spinner("Analyzing..."):
                    # Choose model based on selection
                    model = primary_model if model_choice == "VGG16 Transfer Learning" else backup_model
                    prediction = predict_disease(model, image)
                    
                    # Display results
                    st.success("Analysis Complete!")
                    st.write(f"**Prediction:** {prediction['class']}")
                    st.write(f"**Confidence:** {prediction['confidence']:.2%}")

if __name__ == "__main__":
    main()