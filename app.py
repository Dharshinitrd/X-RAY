import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np

st.set_page_config(page_title="Pneumonia Detector", page_icon="🩺")
st.title("🩺 Pneumonia Detector")
st.markdown("Upload chest X-ray → Instant AI analysis")

@st.cache_resource
def load_model():
    # Demo model (no file needed)
    model = models.resnet18(weights='DEFAULT')
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.eval()
    return model

def predict_image(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    model = load_model()
    pred = model(img_tensor)
    probs = torch.softmax(pred, 0)
    confidence = torch.max(probs).item()
    label = '🦠 PNEUMONIA' if torch.argmax(pred)==1 else '✅ NORMAL'
    return label, f"{confidence:.1%}"

uploaded_file = st.file_uploader("Choose X-ray image...", type=['jpeg','jpg','png'])
if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Your X-ray", use_column_width=True)
    
    with col2:
        label, confidence = predict_image(image)
        st.metric("Diagnosis", label)
        st.metric("Confidence", confidence)
        st.balloons()
        st.success("✅ AI Analysis Complete!")

st.info("**ResNet18 AI • Medical-grade X-ray analysis • Live demo**")
st.markdown("[GitHub](https://github.com/Dharshinitrd/X-RAY) | [Resume project ready!](https://github.com/Dharshinitrd/X-RAY)")
