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
    ImageClip, AudioFileClip, concatenate_videoclips,
    audio_fadein, audio_fadeout
)
import random
import glob
import subprocess
import sys
import shutil

# ─────────────────────────────────────────────────────────────────────────────
# 1. INSTALL FFMPEG (Auto for Streamlit Cloud / Local)
# ─────────────────────────────────────────────────────────────────────────────
def install_ffmpeg():
    if shutil.which("ffmpeg") is None:
        st.warning("Installing FFmpeg (required for video)...")
        try:
            subprocess.run(["apt-get", "update", "-y"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["apt-get", "install", "-y", "ffmpeg"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            st.success("FFmpeg installed!")
        except Exception as e:
            st.error(f"FFmpeg install failed: {e}")
            st.info("Install manually: https://ffmpeg.org/download.html")

# Run only if not in Colab and not already installed
if "google.colab" not in sys.modules:
    install_ffmpeg()

# ─────────────────────────────────────────────────────────────────────────────
# 2. DEFINE AllBirdsVideoMaker (MUST MATCH .pth)
# ─────────────────────────────────────────────────────────────────────────────
class AllBirdsVideoMaker:
    def __init__(self, bird_data, templates):
        self.bird_data = bird_data
        self.templates = templates

    def __call__(self, bird_name):
        if bird_name not in self.bird_data:
            raise ValueError(f"Bird '{bird_name}' not found in dataset.")

        data = self.bird_data[bird_name]
        images = data.get('images', [])
        desc = data.get('description', 'A beautiful bird.')
        colors = data.get('colors', [])

        # Build story
        color_phrase = ", ".join([c.strip() for c in colors]) if colors else "colorful"
        story = random.choice(self.templates).format(
            name=bird_name, color_phrase=color_phrase, desc=desc
        )

        # Generate narration
        audio_file = "narration.mp3"
        try:
            gTTS(story, lang='en').save(audio_file)
        except Exception as e:
            raise RuntimeError(f"TTS failed: {e}")

        audio = AudioFileClip(audio_file)
        narration = audio_fadein(audio, 0.6).audio_fadeout(1.2)

        # Create Ken Burns clips
        clips = []
        duration_per_img = 4.0
        valid_images = [p for p in images if os.path.exists(p)]
        
        if not valid_images:
            raise FileNotFoundError(f"No images found for {bird_name}")

        for img_path in valid_images:
            try:
                clip = ImageClip(img_path).set_duration(duration_per_img)
                clip = clip.resize(lambda t: 1 + 0.15 * (t / duration_per_img))
                clip = clip.fadein(0.3).fadeout(0.3)
                clips.append(clip)
            except Exception as e:
                st.warning(f"Skipping bad image {img_path}: {e}")

        if not clips:
            raise RuntimeError("No valid video clips created.")

        # Combine video
        video = concatenate_videoclips(clips, method="compose")
        total_duration = duration_per_img * len(clips)

        # Sync audio
        if narration.duration > total_duration:
            narration = narration.subclip(0, total_duration)
        else:
            narration = narration.loop(duration=total_duration)

        video = video.set_audio(narration).resize(height=720)

        # Write video
        output_path = f"{bird_name.replace(' ', '_')}_STORY.mp4"
        try:
            video.write_videofile(
                output_path,
                fps=24,
                codec="libx264",
                audio_codec="aac",
                verbose=False,
                logger=None
            )
        except Exception as e:
            raise RuntimeError(f"Video write failed: {e}")

        # Cleanup
        if os.path.exists(audio_file):
            os.remove(audio_file)

        return output_path

# ─────────────────────────────────────────────────────────────────────────────
# 3. LOAD VIDEO MAKER (.pth)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_video_maker():
    path = "all_birds_video_maker.pth"
    if not os.path.exists(path):
        st.error(f"Missing: `{path}` — Upload it to your app folder!")
        return None
    try:
        import torch.serialization
        torch.serialization.add_safe_globals([AllBirdsVideoMaker])
        maker = torch.load(path, map_location="cpu", weights_only=False)
        st.success("Video generator loaded!")
        return maker
    except Exception as e:
        st.error(f"Failed to load .pth: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

video_maker = load_video_maker()

# ─────────────────────────────────────────────────────────────────────────────
# 4. LOAD MODEL & LABEL MAP
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
        else:
            st.error("Model file `resnet34_weights.pth` not found!")
            return None
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

@st.cache_data
def load_label_map():
    try:
        if os.path.exists("label_map.json"):
            with open("label_map.json") as f:
                data = json.load(f)
            return {v: k for k, v in data.items()}
        else:
            st.error("`label_map.json` not found!")
            return {}
    except Exception as e:
        st.error(f"Label map error: {e}")
        return {}

model = load_model()
label_map = load_label_map()

# ─────────────────────────────────────────────────────────────────────────────
# 5. IMAGE PREPROCESS & PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_image(img):
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return t(img).unsqueeze(0)

def predict_species(model, label_map, img):
    if model is None or label_map is None:
        return None
    try:
        x = preprocess_image(img)
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1)
            conf, idx = torch.max(probs, dim=1)
        name = label_map.get(idx.item(), "Unknown Bird")
        return {"species": name, "confidence": conf.item() * 100}
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 6. GENERATE VIDEO (with debug)
# ─────────────────────────────────────────────────────────────────────────────
def generate_bird_video(bird_name: str):
    if video_maker is None:
        st.error("Video maker not loaded.")
        return None
    if not bird_name.strip():
        st.warning("Bird name is empty.")
        return None

    try:
        with st.spinner(f"Creating video for **{bird_name}**..."):
            st.info(f"Birds in dataset: {len(video_maker.bird_data)}")
            path = video_maker(bird_name.strip())
        if path and os.path.exists(path):
            st.success("Video created!")
            return path
        else:
            st.error("Video file was not created.")
            return None
    except Exception as e:
        st.error(f"Video generation failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 7. UI: BACKGROUND & STYLE
# ─────────────────────────────────────────────────────────────────────────────
def set_background(img_path="ugb1.png"):
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(255,255,255,0.92), rgba(255,255,255,0.92)),
                        url("data:image/png;base64,{b64}");
            background-size: cover;
            background-attachment: fixed;
        }}
        .block-container {{
            background: rgba(255,255,255,0.6);
            backdrop-filter: blur(6px);
            border-radius: 12px;
            padding: 1.5rem;
        }}
        </style>
        """, unsafe_allow_html=True)
set_background()

st.markdown('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Poppins:wght@600;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)
st.markdown("""
<style>
.stButton>button { background: #16a34a; color: white; border-radius: 10px; width: 100%; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 8. HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
try:
    logo = Image.open("ugb1.png")
    w, h = logo.size
    logo = logo.resize((max(1, w//2), max(1, h//2)), Image.LANCZOS)
    c1, c2 = st.columns([1, 3])
    with c1:
        st.image(logo, use_container_width=False)
    with c2:
        st.markdown("<h1 style='font-family: Poppins; background: linear-gradient(90deg, #0f172a, #1f2937); -webkit-background-clip: text; color: transparent;'>Birds in Uganda</h1>", unsafe_allow_html=True)
        st.markdown("Identify birds instantly — then watch a **narrated video story**.")
except:
    st.title("Birds in Uganda")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# 9. TABS: UPLOAD & CAMERA
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Upload Image", "Camera"])

# ── UPLOAD TAB ──
with tab1:
    uploaded = st.file_uploader("Upload a bird photo", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        if st.button("Identify Species", key="id_upload", type="primary"):
            with st.spinner("Analyzing..."):
                result = predict_species(model, label_map, img)
            if result:
                st.session_state.upload_result = result
            else:
                st.error("Prediction failed.")

        if "upload_result" in st.session_state:
            r = st.session_state.upload_result
            st.success(f"**{r['species']}** – {r['confidence']:.1f}% confidence")

            if st.button("Generate Video Story", key="gen_upload", type="primary"):
                video_path = generate_bird_video(r['species'])
                if video_path:
                    video_bytes = open(video_path, "rb").read()
                    st.video(video_bytes)
                    st.download_button(
                        "Download Video",
                        video_bytes,
                        file_name=f"{r['species'].replace(' ', '_')}.mp4",
                        mime="video/mp4"
                    )

# ── CAMERA TAB ──
with tab2:
    if st.button("Start Camera", key="start_cam"):
        st.session_state.cam_active = True

    if st.session_state.get("cam_active", False):
        photo = st.camera_input("Take a photo")
        if photo:
            img = Image.open(photo)
            st.image(img, caption="Captured Photo", use_container_width=True)

            if st.button("Identify Species", key="id_cam", type="primary"):
                with st.spinner("Analyzing..."):
                    result = predict_species(model, label_map, img)
                if result:
                    st.session_state.cam_result = result
                else:
                    st.error("Prediction failed.")

            if "cam_result" in st.session_state:
                r = st.session_state.cam_result
                st.success(f"**{r['species']}** – {r['confidence']:.1f}% confidence")

                if st.button("Generate Video Story", key="gen_cam", type="primary"):
                    video_path = generate_bird_video(r['species'])
                    if video_path:
                        video_bytes = open(video_path, "rb").read()
                        st.video(video_bytes)
                        st.download_button(
                            "Download Video",
                            video_bytes,
                            file_name=f"{r['species'].replace(' ', '_')}.mp4",
                            mime="video/mp4"
                        )

        if st.button("Stop Camera", key="stop_cam"):
            st.session_state.cam_active = False

# ─────────────────────────────────────────────────────────────────────────────
# 10. CLEANUP OLD VIDEOS
# ─────────────────────────────────────────────────────────────────────────────
def cleanup():
    for f in glob.glob("*_STORY.mp4") + glob.glob("narration.mp3"):
        try:
            os.remove(f)
        except:
            pass
cleanup()