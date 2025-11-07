# app.py
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
from moviepy.editor import (
    ImageClip, AudioFileClip, concatenate_videoclips
)
from moviepy.audio.fx.all import audio_fadein, audio_fadeout  # FIXED
import random
import glob
import subprocess
import sys
import shutil

# ─────────────────────────────────────────────────────────────────────────────
# 1. AUTO-INSTALL FFMPEG (Streamlit Cloud / Local)
# ─────────────────────────────────────────────────────────────────────────────
def install_ffmpeg():
    if shutil.which("ffmpeg") is None:
        st.warning("FFmpeg not found. Installing...")
        try:
            subprocess.run(["apt-get", "update", "-y"], check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["apt-get", "install", "-y", "ffmpeg"], check=True, stdout=subprocess.DEVNULL)
            st.success("FFmpeg installed!")
        except Exception as e:
            st.error(f"FFmpeg install failed: {e}")

if "google.colab" not in sys.modules:
    install_ffmpeg()

# ─────────────────────────────────────────────────────────────────────────────
# 2. AllBirdsVideoMaker CLASS (MUST MATCH .pth)
# ─────────────────────────────────────────────────────────────────────────────
class AllBirdsVideoMaker:
    def __init__(self, bird_data, templates):
        self.bird_data = bird_data
        self.templates = templates

    def __call__(self, bird_name):
        if bird_name not in self.bird_data:
            raise ValueError(f"Bird '{bird_name}' not found")

        data = self.bird_data[bird_name]
        images = [p for p in data.get('images', []) if os.path.exists(p)]
        desc = data.get('description', 'A beautiful bird.')
        colors = data.get('colors', [])

        color_phrase = ", ".join([c.strip() for c in colors]) if colors else "colorful"
        story = random.choice(self.templates).format(name=bird_name, color_phrase=color_phrase, desc=desc)

        # TTS
        audio_file = "narration.mp3"
        try:
            gTTS(story, lang='en').save(audio_file)
        except Exception as e:
            raise RuntimeError(f"TTS failed: {e}")

        audio = AudioFileClip(audio_file)
        narration = audio_fadein(audio, 0.6).audio_fadeout(1.2)

        # Ken Burns clips
        clips = []
        duration = 4.0
        for img_path in images:
            try:
                clip = ImageClip(img_path).set_duration(duration)
                clip = clip.resize(lambda t: 1 + 0.15 * (t / duration))
                clip = clip.fadein(0.3).fadeout(0.3)
                clips.append(clip)
            except Exception as e:
                st.warning(f"Skipping image {img_path}: {e}")

        if not clips:
            raise RuntimeError("No valid images")

        video = concatenate_videoclips(clips, method="compose")
        total_dur = duration * len(clips)

        if narration.duration > total_dur:
            narration = narration.subclip(0, total_dur)
        else:
            narration = narration.loop(duration=total_dur)

        video = video.set_audio(narration).resize(height=720)
        output = f"{bird_name.replace(' ', '_')}_STORY.mp4"

        try:
            video.write_videofile(output, fps=24, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        except Exception as e:
            raise RuntimeError(f"Video write failed: {e}")

        if os.path.exists(audio_file):
            os.remove(audio_file)

        return output

# ─────────────────────────────────────────────────────────────────────────────
# 3. LOAD .pth
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_video_maker():
    path = "all_birds_video_maker.pth"
    if not os.path.exists(path):
        st.error(f"Missing: `{path}` — Upload it!")
        return None
    try:
        import torch.serialization
        torch.serialization.add_safe_globals([AllBirdsVideoMaker])
        maker = torch.load(path, map_location="cpu", weights_only=False)
        st.success("Video generator loaded!")
        return maker
    except Exception as e:
        st.error(f"Load failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

video_maker = load_video_maker()

# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL & LABEL MAP
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 30)
        if os.path.exists("resnet34_weights.pth"):
            model.load_state_dict(torch.load("resnet34_weights.pth", map_location="cpu"))
            model.eval()
            return model
        st.error("`resnet34_weights.pth` missing!")
        return None
    except Exception as e:
        st.error(f"Model error: {e}")
        return None

@st.cache_data
def load_label_map():
    try:
        if os.path.exists("label_map.json"):
            with open("label_map.json") as f:
                data = json.load(f)
            return {v: k for k, v in data.items()}
        st.error("`label_map.json` missing!")
        return {}
    except Exception as e:
        st.error(f"Label error: {e}")
        return {}

model = load_model()
label_map = load_label_map()

# ─────────────────────────────────────────────────────────────────────────────
# 5. PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_image(img):
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if img.mode != 'RGB': img = img.convert('RGB')
    return t(img).unsqueeze(0)

def predict_species(model, label_map, img):
    if not model or not label_map:
        return None
    try:
        x = preprocess_image(img)
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1)
            conf, idx = torch.max(probs, dim=1)
        name = label_map.get(idx.item(), "Unknown")
        return {"species": name, "confidence": conf.item() * 100}
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 6. VIDEO GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_bird_video(bird_name: str):
    if not video_maker:
        st.error("Video maker not loaded.")
        return None
    try:
        with st.spinner(f"Creating video for **{bird_name}**..."):
            path = video_maker(bird_name.strip())
        if path and os.path.exists(path):
            st.success("Video ready!")
            return path
        else:
            st.error("Video file not created.")
            return None
    except Exception as e:
        st.error(f"Video error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 7. UI
# ─────────────────────────────────────────────────────────────────────────────
def set_bg(img_path="ugb1.png"):
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <style>
        .stApp {{ background: linear-gradient(rgba(255,255,255,0.92), rgba(255,255,255,0.92)), url("data:image/png;base64,{b64}"); background-size: cover; }}
        .block-container {{ background: rgba(255,255,255,0.6); backdrop-filter: blur(6px); border-radius: 12px; padding: 1.5rem; }}
        </style>
        """, unsafe_allow_html=True)
set_bg()

st.markdown('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Poppins:wght@600;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)

try:
    logo = Image.open("ugb1.png")
    w, h = logo.size
    logo = logo.resize((max(1, w//2), max(1, h//2)), Image.LANCZOS)
    c1, c2 = st.columns([1, 3])
    with c1: st.image(logo, use_container_width=False)
    with c2:
        st.markdown("<h1 style='font-family: Poppins; background: linear-gradient(90deg, #0f172a, #1f2937); -webkit-background-clip: text; color: transparent;'>Birds in Uganda</h1>", unsafe_allow_html=True)
        st.markdown("Identify → Generate narrated video story")
except: st.title("Birds in Uganda")

st.markdown("---")

tab1, tab2 = st.tabs(["Upload", "Camera"])

# UPLOAD
with tab1:
    uploaded = st.file_uploader("Upload bird photo", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded", use_container_width=True)
        if st.button("Identify", key="id_up", type="primary"):
            with st.spinner("Analyzing..."):
                res = predict_species(model, label_map, img)
            if res:
                st.session_state.upload_res = res

        if "upload_res" in st.session_state:
            r = st.session_state.upload_res
            st.success(f"**{r['species']}** – {r['confidence']:.1f}%")
            if st.button("Generate Video", key="gen_up", type="primary"):
                path = generate_bird_video(r['species'])
                if path:
                    st.video(path)
                    with open(path, "rb") as f:
                        st.download_button("Download", f, f"{r['species'].replace(' ', '_')}.mp4", "video/mp4")

# CAMERA
with tab2:
    if st.button("Start Camera", key="cam_start"):
        st.session_state.cam_active = True
    if st.session_state.get("cam_active"):
        photo = st.camera_input("Take photo")
        if photo:
            img = Image.open(photo)
            st.image(img, caption="Captured", use_container_width=True)
            if st.button("Identify", key="id_cam", type="primary"):
                with st.spinner("Analyzing..."):
                    res = predict_species(model, label_map, img)
                if res:
                    st.session_state.cam_res = res

            if "cam_res" in st.session_state:
                r = st.session_state.cam_res
                st.success(f"**{r['species']}** – {r['confidence']:.1f}%")
                if st.button("Generate Video", key="gen_cam", type="primary"):
                    path = generate_bird_video(r['species'])
                    if path:
                        st.video(path)
                        with open(path, "rb") as f:
                            st.download_button("Download", f, f"{r['species'].replace(' ', '_')}.mp4", "video/mp4")
        if st.button("Stop Camera", key="cam_stop"):
            st.session_state.cam_active = False

# CLEANUP
def cleanup():
    for f in glob.glob("*_STORY.mp4") + glob.glob("narration.mp3"):
        try: os.remove(f)
        except: pass
cleanup()