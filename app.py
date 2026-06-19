import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import urllib.request
import os

# --- UI CONFIGURATION ---
st.set_page_config(page_title="MRI Classification Engine", page_icon="🧠")
st.title("Brain Tumor MRI Classification Engine")
st.write("Upload an MRI scan to securely process it through our fine-tuned ResNet-18 architecture.")

# --- ARCHITECTURE INITIALIZATION ---
@st.cache_resource
def load_model():
    weights_path = "resnet18_tumor_weights_v2.pth"
    # Pointing directly to your specific GitHub Release vault
    url = "https://github.com/Sridhax-2995/Brain-Tumor-ML/releases/download/v1.0/resnet18_tumor_weights_v2.pth"
    
    # Download the weights if they aren't already loaded into the server
    if not os.path.exists(weights_path):
        with st.spinner("Downloading clinical weights from vault (47MB)..."):
            urllib.request.urlretrieve(url, weights_path)
    
    # Rebuild the exact ResNet-18 structure
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # Load the memory into the structure
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# --- PRE-PROCESSING PIPELINE ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- INTERACTIVE INFERENCE ---
uploaded_file = st.file_uploader("Upload Medical Image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Target MRI Scan", width=300)
    
    st.write("Executing forward pass...")
    tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        confidence, predicted_class = torch.max(probabilities, 0)
        
    prediction = "YES (Tumor Detected)" if predicted_class.item() == 1 else "NO (Healthy Brain)"
    conf_score = confidence.item() * 100
    
    st.divider()
    st.markdown(f"### 🧬 Diagnostic Prediction: **{prediction}**")
    st.markdown(f"### 📊 Algorithmic Confidence: **{conf_score:.2f}%**")
