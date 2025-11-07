import streamlit as st
from PIL import Image
import base64
import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from gtts import gTTS
from moviepy.editor import *
from moviepy.audio.fx.all import audio_fadein, audio_fadeout
import tempfile
import random

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load Video Maker (.pth) – MUST BE IN SAME FOLDER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_video_maker():
    if not os.path.exists("all_birds_video_maker.pth"):
        st.error("`all_birds_video_maker.pth` not found! Upload it to your app folder.")
        return None
    return torch.load("all_birds_video_maker.pth", map_location="cpu")

video_maker = load_video_maker()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Your Existing Functions (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def _set_background_glass(img_path: str = "ugb1.png"):
    try:
        if not os.path.exists(img_path): return
        with open(img_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        css = f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(255,255,255,0.92), rgba(255,255,255,0.92)), url("data:image/png;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .stApp .main .block-container {{
            background: rgba(255,255,255,0.6);
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
            border-radius: 12px;
            padding: 1rem 1.5rem;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except: pass

_set_background_glass("ugb1.png")

# Model loading
@st.cache_resource
def load_model():
    try:
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 30)
        if os.path.exists("resnet34_weights.pth"):
            model.load_state_dict(torch.load("resnet34_weights.pth", map_location="cpu"))
            model.eval()
            return model
        else:
            st.error("Model file not found: resnet34_weights.pth")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_label_map():
    try:
        if os.path.exists("label_map.json"):
            with open("label_map.json", "r") as f:
                label_map = json.load(f)
            return {v: k for k, v in label_map.items()}
        else:
            st.error("Label map file not found: label_map.json")
            return None
    except Exception as e:
        st.error(f"Error loading label map: {e}")
        return None

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if image.mode != 'RGB': image = image.convert('RGB')
    return transform(image).unsqueeze(0)

def predict_species(model, label_map, image):
    try:
        tensor = preprocess_image(image)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_idx = torch.max(probs, dim=1)
        idx = top_idx.item()
        prob = top_prob.item()
        bird_name = label_map.get(idx, f"Class {idx}")
        return {'species': bird_name, 'confidence': prob * 100}
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Load model & map
model = load_model()
label_map = load_label_map()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Video Generation Function (using .pth)
# ─────────────────────────────────────────────────────────────────────────────
def generate_bird_video(bird_name):
    if video_maker is None:
        st.error("Video maker not loaded.")
        return None

    try:
        with st.spinner(f"Generating video for **{bird_name}**..."):
            video_path = video_maker(bird_name)
        return video_path
    except Exception as e:
        st.error(f"Video generation failed: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 4. UI: Your Beautiful Design (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)
st.markdown("""<style>
/* [Your full CSS from before – paste it here] */
:root { --bg: #ffffff; --card: #ffffff; --muted: #1f2937; --text: #0f172a; --brand: #16a34a; --brand-2: #0e7490; --brand-3: #6d28d9; --ring: rgba(15,23,42,0.25); }
html, body, .stApp { font-family: Inter, system-ui; color: var(--text); }
.stApp::before { content: ""; position: fixed; inset: auto auto 10% -10%; width: 40vw; height: 40vw; background: radial-gradient(closest-side, rgba(34,197,94,0.18), transparent 65%); filter: blur(40px); z-index: 0; pointer-events: none; }
.stApp::after { inset: -15% -10% auto auto; width: 35vw; height: 35vw; background: radial-gradient(closest-side, rgba(6,182,212,0.16), transparent 65%); }
.hero { background: linear-gradient(145deg, rgba(15,23,42,0.9), rgba(15,23,42,0.55)); border: 1px solid rgba(255,255,255,0.06); border-radius: 20px; padding: 2.25rem; margin: 0.75rem 0 1.5rem 0; box-shadow: 0 12px 30px rgba(2,6,23,0.45); }
.hero-title { font-family: Poppins; font-weight: 800; letter-spacing: -0.02em; font-size: clamp(1.8rem, 2.5vw + 1.2rem, 3.25rem); margin: 0 0 .35rem 0; background: linear-gradient(90deg, #0f172a, #1f2937 45%, #0b1220 85%); -webkit-background-clip: text; color: transparent; }
.hero-sub { color: var(--muted); font-size: 1.05rem; line-height: 1.6; }
.badge { font-size: .8rem; color: #0f172a; background: rgba(15,23,42,0.08); padding: .35rem .6rem; border-radius: 999px; border: 1px solid rgba(15,23,42,0.15); font-weight: 600; }
.stButton > button { background: linear-gradient(135deg, var(--brand), #16a34a); color: white; border: 0; padding: .7rem 1rem; border-radius: 10px; width: 100%; font-weight: 600; box-shadow: 0 6px 14px rgba(16,185,129,0.28); transition: all .2s; }
.stButton > button:hover { filter: brightness(1.05); box-shadow: 0 10px 18px rgba(16,185,129,0.32); }
.result-card { background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%); border: 2px solid rgba(16,185,129,0.2); border-radius: 12px; padding: 1.25rem; margin: 1rem 0; box-shadow: 0 4px 12px rgba(16,185,129,0.1); }
.result-species { color: #0f172a; font-size: 1.1rem; font-weight: 600; }
.result-confidence { color: #16a34a; font-size: 0.95rem; font-weight: 500; }
</style>""", unsafe_allow_html=True)

# Logo + Hero
try:
    _logo = Image.open("ugb1.png")
    _w, _h = _logo.size
    _new_w, _new_h = max(1, _w // 2), max(1, _h // 2)
    _logo_small = _logo.resize((_new_w, _new_h), Image.LANCZOS)
    with st.container():
        col1, col2 = st.columns([1, 3])
        with col1: st.image(_logo_small, use_column_width=False)
        with col2:
            st.markdown("<div class='hero-title'>Birds in Uganda</div>", unsafe_allow_html=True)
            st.markdown("<div class='hero-sub'>Identify birds instantly. See a narrated video story of your bird.</div>", unsafe_allow_html=True)
except: pass

# ─────────────────────────────────────────────────────────────────────────────
# 5. Tabs: Upload & Camera
# ─────────────────────────────────────────────────────────────────────────────
tab_upload, tab_camera = st.tabs(["Upload", "Camera"])

with tab_upload:
    uploaded_file = st.file_uploader("Upload bird image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Identify & Generate Video", key="upload_btn"):
            if model and label_map:
                with st.spinner("Predicting..."):
                    result = predict_species(model, label_map, image)
                if result:
                    st.session_state.predicted_bird = result['species']
                    st.session_state.predicted_image = image
                else:
                    st.error("Prediction failed.")
            else:
                st.error("Model not loaded.")

        if 'predicted_bird' in st.session_state:
            bird = st.session_state.predicted_bird
            st.markdown(f"### {bird}")
            st.markdown(f"**Confidence:** {st.session_state.get('confidence', 0):.1f}%")

            if st.button("Generate Video Story", key="gen_video_upload"):
                video_path = generate_bird_video(bird)
                if video_path and os.path.exists(video_path):
                    video_bytes = open(video_path, "rb").read()
                    st.video(video_bytes)
                    st.download_button("Download Video", video_bytes, file_name=f"{bird}.mp4", mime="video/mp4")
                else:
                    st.error("Video generation failed.")

with tab_camera:
    camera_photo = st.camera_input("Take a photo")
    if camera_photo:
        image = Image.open(camera_photo)
        st.image(image, caption="Captured Photo", use_column_width=True)
        
        if st.button("Identify & Generate Video", key="camera_btn"):
            if model and label_map:
                with st.spinner("Predicting..."):
                    result = predict_species(model, label_map, image)
                if result:
                    st.session_state.predicted_bird = result['species']
                    st.session_state.predicted_image = image
            else:
                st.error("Model not loaded.")

        if 'predicted_bird' in st.session_state:
            bird = st.session_state.predicted_bird
            st.markdown(f"### {bird}")

            if st.button("Generate Video Story", key="gen_video_camera"):
                video_path = generate_bird_video(bird)
                if video_path and os.path.exists(video_path):
                    video_bytes = open(video_path, "rb").read()
                    st.video(video_bytes)
                    st.download_button("Download Video", video_bytes, file_name=f"{bird}.mp4", mime="video/mp4")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About")
    st.markdown("Upload or snap a photo → get **instant ID + narrated video story** of your bird.")
    st.markdown("Powered by ResNet34 + `all_birds_video_maker.pth`")

st.markdown("<div style='text-align:center; color:#334155; margin-top:2rem; font-size:.9rem;'>Built with for Uganda's Birds</div>", unsafe_allow_html=True)