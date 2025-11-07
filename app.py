# app.py
import streamlit as st
from PIL import Image
import base64, os, json, torch, torch.nn as nn, random, glob, subprocess, sys, shutil
from torchvision import transforms, models
from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from moviepy.audio.fx.all import audio_fadein, audio_fadeout   # FIXED

# ─────────────────────────────────────────────────────────────────────────────
# 1. INSTALL FFMPEG (auto)
# ─────────────────────────────────────────────────────────────────────────────
def install_ffmpeg():
    if shutil.which("ffmpeg") is None:
        st.warning("Installing FFmpeg…")
        try:
            subprocess.run(["apt-get", "update", "-y"], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["apt-get", "install", "-y", "ffmpeg"], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            st.success("FFmpeg ready")
        except Exception as e:
            st.error(f"FFmpeg failed: {e}")

if "google.colab" not in sys.modules:
    install_ffmpeg()

# ─────────────────────────────────────────────────────────────────────────────
# 2. AllBirdsVideoMaker (matches .pth)
# ─────────────────────────────────────────────────────────────────────────────
class AllBirdsVideoMaker:
    def __init__(self, bird_data, templates):
        self.bird_data = bird_data
        self.templates = templates

    def __call__(self, bird_name):
        if bird_name not in self.bird_data:
            raise ValueError(f"Bird '{bird_name}' not found")

        data = self.bird_data[bird_name]
        b64_list = data.get('images_b64', [])
        desc = data.get('description', 'A beautiful bird.')
        colors = data.get('colors', [])

        color_phrase = ", ".join([c.strip() for c in colors]) if colors else "colorful"
        story = random.choice(self.templates).format(
            name=bird_name, color_phrase=color_phrase, desc=desc)

        # TTS
        audio_file = "narration.mp3"
        gTTS(story, lang='en').save(audio_file)
        audio = AudioFileClip(audio_file)
        narration = audio_fadein(audio, 0.6).audio_fadeout(1.2)

        # Decode & Ken Burns
        clips = []
        dur = 4.0
        for b64_str in b64_list:
            img_data = base64.b64decode(b64_str)
            tmp_path = f"tmp_{random.randint(0,9999)}.jpg"
            with open(tmp_path, "wb") as f:
                f.write(img_data)
            clip = ImageClip(tmp_path).set_duration(dur)
            clip = clip.resize(lambda t: 1 + 0.15*(t/dur))
            clip = clip.fadein(0.3).fadeout(0.3)
            clips.append(clip)

        if not clips:
            raise RuntimeError("No images decoded")

        video = concatenate_videoclips(clips, method="compose")
        total = dur * len(clips)

        if narration.duration > total:
            narration = narration.subclip(0, total)
        else:
            narration = narration.loop(duration=total)

        video = video.set_audio(narration).resize(height=720)
        out = f"{bird_name.replace(' ', '_')}_STORY.mp4"
        video.write_videofile(out, fps=24, codec="libx264", audio_codec="aac",
                              verbose=False, logger=None)

        # cleanup
        os.remove(audio_file)
        for f in glob.glob("tmp_*.jpg"):
            try: os.remove(f)
            except: pass
        return out

# ─────────────────────────────────────────────────────────────────────────────
# 3. LOAD .pth
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_video_maker():
    p = "all_birds_video_maker.pth"
    if not os.path.exists(p):
        st.error(f"Missing `{p}` – upload the **new** .pth")
        return None
    try:
        import torch.serialization
        torch.serialization.add_safe_globals([AllBirdsVideoMaker])
        maker = torch.load(p, map_location="cpu", weights_only=False)
        st.success("Video generator loaded")
        return maker
    except Exception as e:
        st.error(f"Load error: {e}")
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
        m = models.resnet34(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 30)
        if os.path.exists("resnet34_weights.pth"):
            m.load_state_dict(torch.load("resnet34_weights.pth", map_location="cpu"))
            m.eval()
            return m
        st.error("`resnet34_weights.pth` missing")
        return None
    except Exception as e:
        st.error(f"Model error: {e}")
        return None

@st.cache_data
def load_label_map():
    try:
        if os.path.exists("label_map.json"):
            with open("label_map.json") as f:
                d = json.load(f)
            return {v: k for k, v in d.items()}
        st.error("`label_map.json` missing")
        return {}
    except Exception as e:
        st.error(f"Label error: {e}")
        return {}

model = load_model()
label_map = load_label_map()

# ─────────────────────────────────────────────────────────────────────────────
# 5. PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(img):
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if img.mode != 'RGB': img = img.convert('RGB')
    return t(img).unsqueeze(0)

def predict_species(m, lm, img):
    if not m or not lm: return None
    try:
        x = preprocess(img)
        with torch.no_grad():
            out = m(x)
            prob = torch.softmax(out, dim=1)
            conf, idx = torch.max(prob, 1)
        name = lm.get(idx.item(), "Unknown")
        return {"species": name, "confidence": conf.item()*100}
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 6. VIDEO GENERATION (debug)
# ─────────────────────────────────────────────────────────────────────────────
def generate_bird_video(bird_name: str):
    if not video_maker:
        st.error("Video maker not loaded")
        return None
    try:
        with st.spinner(f"Creating video for **{bird_name}**…"):
            path = video_maker(bird_name.strip())
        if path and os.path.exists(path):
            st.success("Video ready")
            return path
        else:
            st.error("File not created")
            return None
    except Exception as e:
        st.error(f"Video error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 7. UI
# ─────────────────────────────────────────────────────────────────────────────
def set_bg(p="ugb1.png"):
    if os.path.exists(p):
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <style>
        .stApp {{ background: linear-gradient(rgba(255,255,255,0.92), rgba(255,255,255,0.92)),
                        url("data:image/png;base64,{b64}"); background-size: cover; }}
        .block-container {{ background: rgba(255,255,255,0.6); backdrop-filter: blur(6px);
                           border-radius: 12px; padding: 1.5rem; }}
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
        st.markdown("<h1 style='font-family: Poppins; background: linear-gradient(90deg,#0f172a,#1f2937); -webkit-background-clip:text; color:transparent;'>Birds in Uganda</h1>", unsafe_allow_html=True)
        st.markdown("Identify → **Narrated video story**")
except: st.title("Birds in Uganda")

st.markdown("---")

tab1, tab2 = st.tabs(["Upload", "Camera"])

# ── UPLOAD ──
with tab1:
    up = st.file_uploader("Upload bird photo", type=["png","jpg","jpeg"])
    if up:
        img = Image.open(up)
        st.image(img, caption="Uploaded", use_container_width=True)
        if st.button("Identify", key="id_up", type="primary"):
            with st.spinner("Analyzing…"):
                res = predict_species(model, label_map, img)
            if res: st.session_state.upload_res = res

        if "upload_res" in st.session_state:
            r = st.session_state.upload_res
            st.success(f"**{r['species']}** – {r['confidence']:.1f}%")
            if st.button("Generate Video", key="gen_up", type="primary"):
                p = generate_bird_video(r['species'])
                if p:
                    st.video(p)
                    with open(p, "rb") as f:
                        st.download_button("Download", f, f"{r['species'].replace(' ','_')}.mp4", "video/mp4")

# ── CAMERA ──
with tab2:
    if st.button("Start Camera", key="cam_start"):
        st.session_state.cam_active = True
    if st.session_state.get("cam_active"):
        photo = st.camera_input("Take photo")
        if photo:
            img = Image.open(photo)
            st.image(img, caption="Captured", use_container_width=True)
            if st.button("Identify", key="id_cam", type="primary"):
                with st.spinner("Analyzing…"):
                    res = predict_species(model, label_map, img)
                if res: st.session_state.cam_res = res

            if "cam_res" in st.session_state:
                r = st.session_state.cam_res
                st.success(f"**{r['species']}** – {r['confidence']:.1f}%")
                if st.button("Generate Video", key="gen_cam", type="primary"):
                    p = generate_bird_video(r['species'])
                    if p:
                        st.video(p)
                        with open(p, "rb") as f:
                            st.download_button("Download", f, f"{r['species'].replace(' ','_')}.mp4", "video/mp4")
        if st.button("Stop Camera", key="cam_stop"):
            st.session_state.cam_active = False

# ── CLEANUP ──
def cleanup():
    for f in glob.glob("*_STORY.mp4") + glob.glob("narration.mp3") + glob.glob("tmp_*.jpg"):
        try: os.remove(f)
        except: pass
cleanup()