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
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
import tempfile
import random

# Display logo (centered and resized to one-quarter of original dimensions)
def _set_background_glass(img_path: str = "ugb1.png"):
    """Set a full-page background using the given image and add a translucent glass
    style to the main Streamlit block container so content appears on a frosted panel.
    The image is embedded as a data URI to improve compatibility when deployed.
    """
    try:
        if not os.path.exists(img_path):
            return
        with open(img_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        css = f"""
        <style>
        .stApp {{
            /* Apply a white overlay so the image appears very subtle, requiring focus to notice */
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
    except Exception:
        # If embedding fails, don't break the app
        pass

# Apply the background/glass style
_set_background_glass("ugb1.png")

# Model loading and prediction functions
@st.cache_resource
def load_model():
    """Load the ResNet34 model with trained weights"""
    try:
        model = models.resnet34(weights=None)
        num_classes = 30  # Based on label_map.json having 30 classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Load weights
        if os.path.exists("resnet34_weights.pth"):
            model.load_state_dict(torch.load("resnet34_weights.pth", map_location=torch.device('cpu')))
            model.eval()
            return model
        else:
            st.error("Model file not found: resnet34_weights.pth")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def load_label_map():
    """Load the label mapping from JSON file"""
    try:
        if os.path.exists("label_map.json"):
            with open("label_map.json", "r") as f:
                label_map = json.load(f)
            # Reverse mapping: index -> bird name
            idx_to_label = {v: k for k, v in label_map.items()}
            return idx_to_label
        else:
            st.error("Label map file not found: label_map.json")
            return None
    except Exception as e:
        st.error(f"Error loading label map: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def predict_species(model, label_map, image):
    """Predict bird species from image - returns top prediction only"""
    try:
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)  # Shape: (batch_size, num_classes) = (1, 30)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Apply softmax along class dimension
            
        # Get top prediction (dim=1 is the class dimension)
        top_prob, top_index = torch.max(probabilities, dim=1)
        
        idx = top_index.item()
        prob = top_prob.item()
        bird_name = label_map.get(idx, f"Class {idx}")
        
        result = {
            'species': bird_name,
            'confidence': prob * 100
        }
        
        return result
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

# Load model and label map
model = load_model()
label_map = load_label_map()

# Bird descriptions and colors (simplified - you can expand this)
BIRD_DATA = {
    "African Fish-Eagle": {
        "description": "A majestic bird of prey with a distinctive white head and brown body. It is known for its powerful call and excellent fishing abilities.",
        "colors": ["white", "brown", "black"]
    },
    "African Sacred Ibis": {
        "description": "A wading bird with a long curved bill and black and white plumage. It is often seen near water bodies.",
        "colors": ["white", "black"]
    },
    "Black Kite": {
        "description": "A medium-sized bird of prey with dark brown plumage. It is an agile flyer and often seen soaring in the sky.",
        "colors": ["brown", "black"]
    },
    "Grey Heron": {
        "description": "A tall wading bird with grey plumage and a long neck. It stands motionless waiting for fish in shallow water.",
        "colors": ["grey", "white", "black"]
    },
    "Lilac-breasted Roller": {
        "description": "A stunning bird with vibrant colors including lilac, blue, green, and brown. It is known for its acrobatic flight displays.",
        "colors": ["lilac", "blue", "green", "brown"]
    },
    "Pied Kingfisher": {
        "description": "A black and white kingfisher with a distinctive hovering flight pattern. It dives into water to catch fish.",
        "colors": ["black", "white"]
    }
}

def get_bird_info(bird_name):
    """Get bird information or return default"""
    return BIRD_DATA.get(bird_name, {
        "description": f"The {bird_name} is a beautiful bird found in Uganda. It adds to the rich biodiversity of the region.",
        "colors": ["various"]
    })

def generate_vivid_bird_story(bird_name, description, colors):
    """Generate a vivid story about the bird"""
    color_phrase = ", ".join(colors) if colors else "beautiful"
    desc = description.strip().capitalize()
    
    templates = [
        f"The {bird_name} is a beautiful bird with {color_phrase} feathers. {desc} It moves gracefully and brings joy to anyone who sees it.",
        f"In the trees, the {bird_name} stands out with its {color_phrase} colors. {desc} It sings softly and watches the world with calm eyes.",
        f"The {bird_name} is easy to spot because of its bright {color_phrase} feathers. {desc} It flies quickly and loves to explore the forest.",
        f"With {color_phrase} feathers, the {bird_name} looks like a flying rainbow. {desc} It's a peaceful bird that enjoys the quiet of nature.",
        f"The {bird_name} is known for its {color_phrase} colors and gentle sounds. {desc} It glides through the air like a leaf in the wind."
    ]
    return random.choice(templates)

def generate_video(bird_name, image_path, output_path):
    """Generate a video with narration about the bird"""
    try:
        # Get bird information
        bird_info = get_bird_info(bird_name)
        story = generate_vivid_bird_story(bird_name, bird_info["description"], bird_info["colors"])
        
        # Generate audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_audio:
            audio_path = tmp_audio.name
            tts = gTTS(text=story, lang='en', slow=False)
            tts.save(audio_path)
        
        # Create video
        audio = AudioFileClip(audio_path)
        duration = audio.duration
        
        # Create image clip
        img_clip = ImageClip(image_path).set_duration(duration).resize(height=720)
        
        # Add captions (optional - may fail if ImageMagick not available)
        caption_clips = []
        try:
            words = story.split()
            total_words = len(words)
            avg_time_per_word = duration / total_words
            
            i = 0
            while i < total_words:
                remaining = total_words - i
                chunk_size = random.randint(2, min(6, remaining)) if remaining >= 2 else remaining
                chunk_words = words[i:i+chunk_size]
                chunk_text = " ".join(chunk_words)
                chunk_start = i * avg_time_per_word
                chunk_duration = chunk_size * avg_time_per_word
                
                txt_clip = TextClip(
                    chunk_text,
                    fontsize=40,
                    font='Arial-Bold',
                    color='white',
                    stroke_color='black',
                    stroke_width=2,
                    method='caption',
                    size=(img_clip.w, None),
                    align='center'
                )
                txt_clip = txt_clip.set_position(('center', 'bottom')).set_start(chunk_start).set_duration(chunk_duration)
                caption_clips.append(txt_clip)
                
                i += chunk_size
        except Exception as e:
            # If TextClip fails (ImageMagick not available), continue without captions
            pass
        
        # Combine video and audio
        if caption_clips:
            final = CompositeVideoClip([img_clip.set_audio(audio)] + caption_clips)
        else:
            final = img_clip.set_audio(audio)
        
        final.write_videofile(output_path, fps=24, codec='libx264', audio_codec='aac', verbose=False, logger=None)
        
        # Cleanup
        try:
            os.unlink(audio_path)
            audio.close()
            final.close()
            img_clip.close()
        except:
            pass
        
        return output_path
    except Exception as e:
        st.error(f"Error generating video: {str(e)}")
        return None

# Global modern theme: fonts, colors, animations, components polish
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)
st.markdown("""<style>
:root {
  --bg: #ffffff;
  --card: #ffffff;
  --muted: #1f2937; /* slate-800 for strong contrast on light bg */
  --text: #0f172a;  /* slate-900 as primary text */
  --brand: #16a34a;
  --brand-2: #0e7490;
  --brand-3: #6d28d9;
  --ring: rgba(15,23,42,0.25);
}
html, body, .stApp { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica Neue, Arial, "Apple Color Emoji", "Segoe UI Emoji"; color: var(--text); }
.stApp::before, .stApp::after {
  content: "";
  position: fixed;
  inset: auto auto 10% -10%;
  width: 40vw;
  height: 40vw;
  background: radial-gradient(closest-side, rgba(34,197,94,0.18), transparent 65%);
  filter: blur(40px);
  z-index: 0;
  pointer-events: none;
}
.stApp::after {
  inset: -15% -10% auto auto;
  width: 35vw;
  height: 35vw;
  background: radial-gradient(closest-side, rgba(6,182,212,0.16), transparent 65%);
}
.stApp .main .block-container { position: relative; z-index: 1; }
.hero {
  background: linear-gradient(145deg, rgba(15,23,42,0.9), rgba(15,23,42,0.55));
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 20px;
  padding: 2.25rem;
  margin: 0.75rem 0 1.5rem 0;
  box-shadow: 0 12px 30px rgba(2,6,23,0.45);
}
.hero-title {
  font-family: Poppins, Inter, system-ui;
  font-weight: 800;
  letter-spacing: -0.02em;
  font-size: clamp(1.8rem, 2.5vw + 1.2rem, 3.25rem);
  margin: 0 0 .35rem 0;
  background: linear-gradient(90deg, #0f172a, #1f2937 45%, #0b1220 85%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.hero-sub {
  color: var(--muted);
  font-size: 1.05rem;
  line-height: 1.6;
}
.hero-badges { display: flex; gap: .5rem; flex-wrap: wrap; margin-top: .85rem; }
.badge {
  font-size: .8rem;
  color: #0f172a;
  background: rgba(15,23,42,0.08);
  padding: .35rem .6rem;
  border-radius: 999px;
  border: 1px solid rgba(15,23,42,0.15);
  font-weight: 600;
}
.card {
  background: #ffffff;
  border: 1px solid rgba(2,6,23,0.08);
  border-radius: 16px;
  padding: 1.25rem;
  box-shadow: 0 10px 24px rgba(2,6,23,0.06);
  color: var(--text);
}
.card h4 { color: var(--text); margin: 0 0 .5rem 0; font-weight: 700; }
.card .hint { color: var(--muted); font-size: .92rem; margin-bottom: .75rem; }
.stButton > button {
  background: linear-gradient(135deg, var(--brand), #16a34a);
  color: white;
  border: 0;
  padding: .7rem 1rem;
  border-radius: 10px;
  width: 100%;
  font-weight: 600;
  box-shadow: 0 6px 14px rgba(16,185,129,0.28);
  transition: transform .08s ease, filter .2s ease, box-shadow .2s ease;
}
.stButton > button:hover { filter: brightness(1.05); box-shadow: 0 10px 18px rgba(16,185,129,0.32); }
.stButton > button:active { transform: translateY(1px); }
[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] {
  border: 1px dashed rgba(2,6,23,0.15);
  background: #f8fafc;
  transition: border-color .2s ease, background .2s ease, box-shadow .2s ease;
  border-radius: 14px;
}
[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"]:hover {
  border-color: rgba(15,23,42,0.45);
  box-shadow: 0 8px 20px rgba(2,6,23,0.08);
  background: #f1f5f9;
}
[data-testid="stFileUploader"] section > div { color: #0f172a !important; }
[data-testid="stFileUploader"] label { color: #0f172a !important; font-weight: 600; }
[data-testid="stCameraInputLabel"] { color: #0f172a !important; font-weight: 600; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
.fade { animation: fadeUp .4s ease both; }
.input-section {
  background: rgba(2,6,23,0.75);
  border-radius: 14px;
  padding: 1rem;
  margin: 0.5rem 0;
  border: 1px solid rgba(255,255,255,0.06);
}
.section-title {
  color: #0f172a;
  font-size: 1.05rem;
  margin-bottom: 0.75rem;
  font-weight: 600;
}
.result-card {
  background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
  border: 2px solid rgba(16,185,129,0.2);
  border-radius: 12px;
  padding: 1.25rem;
  margin: 1rem 0;
  box-shadow: 0 4px 12px rgba(16,185,129,0.1);
}
.result-title {
  color: #0f172a;
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 0.5rem;
}
.result-item {
  background: #ffffff;
  border-left: 4px solid #16a34a;
  padding: 0.75rem 1rem;
  margin: 0.5rem 0;
  border-radius: 8px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.result-species {
  color: #0f172a;
  font-size: 1.1rem;
  font-weight: 600;
  margin-bottom: 0.25rem;
}
.result-confidence {
  color: #16a34a;
  font-size: 0.95rem;
  font-weight: 500;
}
</style>""", unsafe_allow_html=True)
try:
    _logo = Image.open("ugb1.png")
    _w, _h = _logo.size
    # Prevent zero or negative sizes
    _new_w = max(1, _w // 2)  # Changed from 4 to 2 to make image twice as large
    _new_h = max(1, _h // 2)
    _logo_small = _logo.resize((_new_w, _new_h), Image.LANCZOS)

    # Hero header with logo and gradient title
    with st.container():
        logo_col, text_col = st.columns([1, 3])
        with logo_col:
            st.image(_logo_small, use_column_width=False)
        with text_col:
            st.markdown("<div class='hero-title'>Birds in Uganda</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='hero-sub'>Identify birds from photos in seconds. Upload an image or use your camera to discover species, with a beautiful, distraction-free interface.</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='hero-badges'><span class='badge'>Smart Vision</span><span class='badge'>On-device Capture</span><span class='badge'>UG Species Focus</span></div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
except Exception:
    # If logo not found or cannot be opened, skip silently
    pass

# Remove old italic banner to reduce clutter and rely on hero subtitle

# Main content container with modern layout
with st.container():
    # Section prompt
    st.markdown(
        """
        <div style='text-align:center; margin: .5rem 0 1rem;'>
            <p style='color:#0f172a; margin:0; font-weight:700; font-size:1rem; letter-spacing:.01em;'>
                Choose how you want to identify a bird
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_upload, tab_camera = st.tabs(["📁 Upload", "📷 Camera"])

    with tab_upload:
        st.markdown("<h4>📁 Upload Image</h4>", unsafe_allow_html=True)
        st.markdown("<div class='hint'>PNG or JPEG. Clear, close-up shots improve results.</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Select a bird image", type=['png', 'jpg', 'jpeg'], key="uploader_file")
        if uploaded_file is not None:
            # Clear previous result when new image is uploaded
            if 'upload_result' in st.session_state:
                del st.session_state.upload_result
            
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            if st.button("Identify Specie", key="identify_specie_upload_button"):
                if model is not None and label_map is not None:
                    with st.spinner("🔍 Analyzing image..."):
                        result = predict_species(model, label_map, image)
                    
                    if result:
                        # Store result in session state
                        st.session_state.upload_result = result
                        st.session_state.upload_image = image
                    else:
                        st.error("Failed to predict species. Please try again.")
                else:
                    st.error("Model or label map not loaded. Please check if the files exist.")
            
            # Display result if available
            if 'upload_result' in st.session_state and st.session_state.upload_result:
                result = st.session_state.upload_result
                st.markdown("<div class='result-title'>🦅 Identification Result</div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='result-item'>
                    <div class='result-species'>{result['species']}</div>
                    <div class='result-confidence'>Confidence: {result['confidence']:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Generate video button
                if st.button("🎬 Generate Video", key="generate_video_upload", use_container_width=True):
                    if 'upload_image' in st.session_state:
                        with st.spinner("🎬 Generating video with narration..."):
                            # Save image temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_img:
                                img_path = tmp_img.name
                                st.session_state.upload_image.save(img_path)
                            
                            # Generate video
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                                video_path = tmp_video.name
                            
                            video_file = generate_video(result['species'], img_path, video_path)
                            
                            if video_file and os.path.exists(video_file):
                                # Display video
                                with open(video_file, 'rb') as f:
                                    video_bytes = f.read()
                                st.video(video_bytes)
                                
                                # Download button
                                st.download_button(
                                    label="📥 Download Video",
                                    data=video_bytes,
                                    file_name=f"{result['species'].replace(' ', '_')}_story.mp4",
                                    mime="video/mp4"
                                )
                                
                                # Cleanup
                                try:
                                    os.unlink(img_path)
                                    os.unlink(video_path)
                                except:
                                    pass
                            else:
                                st.error("Failed to generate video. Please try again.")
                    else:
                        st.error("Image not found. Please upload an image again.")
                
                st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_camera:
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False

        st.markdown("<h4>📷 Take Picture</h4>", unsafe_allow_html=True)
        if not st.session_state.camera_active:
            try:
                _placeholder_path = "ub2.png"
                if os.path.exists(_placeholder_path):
                    with open(_placeholder_path, "rb") as _f:
                        _data = _f.read()
                    _b64 = base64.b64encode(_data).decode()
                    _img_html = (
                        f"<img src=\"data:image/png;base64,{_b64}\" "
                        "style=\"width:100%; aspect-ratio:4/3; min-height:280px; object-fit:cover; "
                        "border-radius:12px; margin-bottom:0.75rem; box-shadow: inset 0 0 40px rgba(0,0,0,0.6);\"/>")
                    st.markdown(_img_html, unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<div style='width:100%; aspect-ratio:4/3; min-height:280px; background:linear-gradient(180deg,#0b1220,#0b1220 60%, #0f172a); border-radius:12px; margin-bottom:0.75rem; box-shadow: inset 0 0 40px rgba(0,0,0,0.6);'></div>",
                        unsafe_allow_html=True,
                    )
            except Exception:
                st.markdown(
                    "<div style='width:100%; aspect-ratio:4/3; min-height:280px; background:linear-gradient(180deg,#0b1220,#0b1220 60%, #0f172a); border-radius:12px; margin-bottom:0.75rem; box-shadow: inset 0 0 40px rgba(0,0,0,0.6);'></div>",
                    unsafe_allow_html=True,
                )

            def _start_camera():
                st.session_state.camera_active = True

            st.button("Start Camera 📷", key="use_camera_button", on_click=_start_camera)

        if st.session_state.camera_active:
            camera_photo = st.camera_input("Take a photo", key="camera_input")
            if camera_photo is not None:
                # Clear previous result when new photo is captured
                if 'camera_result' in st.session_state:
                    del st.session_state.camera_result
                
                image = Image.open(camera_photo)
                st.image(image, caption='Captured Photo', use_column_width=True)
                
                if st.button("Identify Specie", key="identify_specie_camera_button"):
                    if model is not None and label_map is not None:
                        with st.spinner("🔍 Analyzing image..."):
                            result = predict_species(model, label_map, image)
                        
                        if result:
                            # Store result in session state
                            st.session_state.camera_result = result
                            st.session_state.camera_image = image
                        else:
                            st.error("Failed to predict species. Please try again.")
                    else:
                        st.error("Model or label map not loaded. Please check if the files exist.")
                
                # Display result if available
                if 'camera_result' in st.session_state and st.session_state.camera_result:
                    result = st.session_state.camera_result
                    st.markdown("<div class='result-title'>🦅 Identification Result</div>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class='result-item'>
                        <div class='result-species'>{result['species']}</div>
                        <div class='result-confidence'>Confidence: {result['confidence']:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Generate video button
                    if st.button("🎬 Generate Video", key="generate_video_camera", use_container_width=True):
                        if 'camera_image' in st.session_state:
                            with st.spinner("🎬 Generating video with narration..."):
                                # Save image temporarily
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_img:
                                    img_path = tmp_img.name
                                    st.session_state.camera_image.save(img_path)
                                
                                # Generate video
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                                    video_path = tmp_video.name
                                
                                video_file = generate_video(result['species'], img_path, video_path)
                                
                                if video_file and os.path.exists(video_file):
                                    # Display video
                                    with open(video_file, 'rb') as f:
                                        video_bytes = f.read()
                                    st.video(video_bytes)
                                    
                                    # Download button
                                    st.download_button(
                                        label="📥 Download Video",
                                        data=video_bytes,
                                        file_name=f"{result['species'].replace(' ', '_')}_story.mp4",
                                        mime="video/mp4"
                                    )
                                    
                                    # Cleanup
                                    try:
                                        os.unlink(img_path)
                                        os.unlink(video_path)
                                    except:
                                        pass
                                else:
                                    st.error("Failed to generate video. Please try again.")
                        else:
                            st.error("Image not found. Please capture an image again.")
                    
                    st.markdown("</div>", unsafe_allow_html=True)

            if st.button("Stop Camera ⏹️", key="stop_camera_button", help="Click to stop camera preview"):
                st.session_state.camera_active = False
        st.markdown("</div>", unsafe_allow_html=True)

    # Sidebar info
    with st.sidebar:
        st.markdown("""
        <div class='card fade'>
          <h4>About</h4>
          <div class='hint'>This demo helps identify birds commonly found across Uganda. For best results, ensure good lighting and a clear subject.</div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown(
        """
        <div style='text-align:center; color:#334155; margin-top: 1rem; font-size:.9rem;'>
            Built for the Love of Nature
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # (no wrapper divs to close)
