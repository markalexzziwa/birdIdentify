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
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from moviepy.audio.fx.all import audio_fadein, audio_fadeout
import random
import glob

# ─────────────────────────────────────────────────────────────────────────────
# 1. DEFINE CLASS FIRST (Critical for PyTorch 2.6+)
# ─────────────────────────────────────────────────────────────────────────────
class AllBirdsVideoMaker:
    def __init__(self, bird_data, templates):
        self.bird_data = bird_data
        self.templates = templates

    def __call__(self, bird_name):
        if bird_name not in self.bird_data:
            raise ValueError(f"Bird not found: {bird_name}")
        
        data = self.bird_data[bird_name]
        images = data['images']
        desc = data['description']
        colors = data['colors']
        
        color_phrase = ", ".join([c.strip() for c in colors]) if colors else "vibrant"
        story = random.choice(self.templates).format(name=bird_name, color_phrase=color_phrase, desc=desc)
        
        # TTS
        audio_file = "temp_narration.mp3"
        gTTS(story, lang='en').save(audio_file)
        
        # Audio
        audio = AudioFileClip(audio_file)
        narration = audio_fadein(audio, 0.6).audio_fadeout(1.2)
        
        # Video
        img_duration = 4.0
        total_duration = img_duration * len(images)
        if narration.duration < total_duration:
            loops = int(total_duration / narration.duration) + 1
            narration = concatenate_audioclips([narration] * loops).subclip(0, total_duration)
        else:
            narration = narration.subclip(0, total_duration)
        
        def ken_burns(img_path):
            clip = ImageClip(img_path).set_duration(img_duration)
            clip = clip.resize(lambda t: 1 + 0.15 * (t / img_duration))
            return clip.fadein(0.3).fadeout(0.3)
        
        clips = [ken_burns(img) for img in images]
        video = concatenate_videoclips(clips, method="compose").set_audio(narration)
        video = video.resize(height=720)
        
        output = f"{bird_name.replace(' ', '_')}_VIDEO.mp4"
        video.write_videofile(output, fps=24, codec="libx264", audio_codec="aac")
        
        os.remove(audio_file)
        return output

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD VIDEO MAKER (.pth)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_video_maker():
    path = "all_birds_video_maker.pth"
    if not os.path.exists(path):
        st.error(f"Missing: `{path}` — Upload it to your app folder!")
        return None
    
    try:
        import torch.serialization
        torch.serialization.add_safe_globals([AllBirdsVideoMaker])  # Now class exists!
        maker = torch.load(path, map_location="cpu", weights_only=False)
        st.success("Video generator loaded!")
        return maker
    except Exception as e:
        st.error(f"Failed to load .pth: {e}")
        return None

video_maker = load_video_maker()

# ─────────────────────────────────────────────────────────────────────────────
# 3. BACKGROUND & MODEL FUNCTIONS (unchanged)
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

def generate_bird_video(bird_name: str):
    if video_maker is None:
        st.error("Video maker not loaded.")
        return None
    if not bird_name.strip():
        st.warning("Bird name empty.")
        return None
    try:
        with st.spinner(f"Creating video for **{bird_name}**..."):
            video_path = video_maker(bird_name.strip())
        if video_path and os.path.exists(video_path):
            return video_path
        else:
            st.error("Video file was not created.")
            return None
    except Exception as e:
        st.error(f"Video generation failed: {e}")
        return None

model = load_model()
label_map = load_label_map()

# ─────────────────────────────────────────────────────────────────────────────
# 4. UI: HERO + TABS (Fixed use_container_width)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)
st.markdown("""<style>
/* [Your full CSS — unchanged] */
</style>""", unsafe_allow_html=True)

try:
    _logo = Image.open("ugb1.png")
    _w, _h = _logo.size
    _new_w, _new_h = max(1, _w // 2), max(1, _h // 2)
    _logo_small = _logo.resize((_new_w, _new_h), Image.LANCZOS)
    with st.container():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(_logo_small, use_container_width=False)  # FIXED
        with col2:
            st.markdown("<div class='hero-title'>Birds in Uganda</div>", unsafe_allow_html=True)
            st.markdown("<div class='hero-sub'>Identify birds instantly — then watch a narrated video story.</div>", unsafe_allow_html=True)
except: pass

with st.container():
    st.markdown("<div style='text-align:center; margin: .5rem 0 1rem;'><p style='color:#0f172a; margin:0; font-weight:700; font-size:1rem;'>Choose how you want to identify a bird</p></div>", unsafe_allow_html=True)
    tab_upload, tab_camera = st.tabs(["Upload", "Camera"])

    # ── UPLOAD TAB ──
    with tab_upload:
        uploaded_file = st.file_uploader("Select a bird image", type=['png', 'jpg', 'jpeg'])
        if uploaded_file:
            if 'upload_result' in st.session_state:
                del st.session_state.upload_result
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_container_width=True)  # FIXED

            if st.button("Identify Specie", key="identify_upload"):
                if model and label_map:
                    with st.spinner("Analyzing..."):
                        result = predict_species(model, label_map, image)
                    if result:
                        st.session_state.upload_result = result
                else:
                    st.error("Model not loaded.")

            if 'upload_result' in st.session_state:
                result = st.session_state.upload_result
                st.markdown(f"<div class='result-title'>Identification Result</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='result-item'><div class='result-species'>{result['species']}</div><div class='result-confidence'>Confidence: {result['confidence']:.2f}%</div></div>", unsafe_allow_html=True)

                if st.button("Generate Video Story", key="gen_video_upload", type="primary"):
                    video_path = generate_bird_video(result['species'])
                    if video_path:
                        video_bytes = open(video_path, "rb").read()
                        st.video(video_bytes)
                        st.download_button("Download Video", video_bytes, f"{result['species'].replace(' ', '_')}_STORY.mp4", "video/mp4")

    # ── CAMERA TAB ──
    with tab_camera:
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False

        if not st.session_state.camera_active:
            st.button("Start Camera", key="start_cam", on_click=lambda: st.session_state.update(camera_active=True))
        else:
            camera_photo = st.camera_input("Take a photo", key="cam_input")
            if camera_photo:
                if 'camera_result' in st.session_state:
                    del st.session_state.camera_result
                image = Image.open(camera_photo)
                st.image(image, caption='Captured Photo', use_container_width=True)  # FIXED

                if st.button("Identify Specie", key="identify_cam"):
                    if model and label_map:
                        with st.spinner("Analyzing..."):
                            result = predict_species(model, label_map, image)
                        if result:
                            st.session_state.camera_result = result
                    else:
                        st.error("Model not loaded.")

                if 'camera_result' in st.session_state:
                    result = st.session_state.camera_result
                    st.markdown(f"<div class='result-title'>Identification Result</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='result-item'><div class='result-species'>{result['species']}</div><div class='result-confidence'>Confidence: {result['confidence']:.2f}%</div></div>", unsafe_allow_html=True)

                    if st.button("Generate Video Story", key="gen_video_camera", type="primary"):
                        video_path = generate_bird_video(result['species'])
                        if video_path:
                            video_bytes = open(video_path, "rb").read()
                            st.video(video_bytes)
                            st.download_button("Download Video", video_bytes, f"{result['species'].replace(' ', '_')}_STORY.mp4", "video/mp4")

            st.button("Stop Camera", key="stop_cam", on_click=lambda: st.session_state.update(camera_active=False))

    with st.sidebar:
        st.markdown("### About")
        st.markdown("Identify birds → get a **narrated video story** instantly.")

st.markdown("<div style='text-align:center; color:#334155; margin-top:2rem; font-size:.9rem;'>Built for Uganda's Birds</div>", unsafe_allow_html=True)

# ── CLEANUP OLD VIDEOS ──
def cleanup():
    for f in glob.glob("*_VIDEO.mp4"):
        try: os.remove(f)
        except: pass
cleanup()