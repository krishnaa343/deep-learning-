import os
from pathlib import Path
import subprocess
import tempfile
import streamlit as st
import tensorflow as tf
import numpy as np
import imageio
import imageio_ffmpeg

from utils import (
    load_data,
    load_video,
    load_alignment_text,
    decode_prediction,
    VIDEOS_DIR,
    ALIGNMENTS_DIR,
    PROJECT_ROOT
)
from modelutil import load_model

# Page configuration
st.set_page_config(
    page_title="Silent Speech Lip-Reading AI",
    page_icon="👄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .gt-box {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 1.15rem;
        font-weight: 600;
        color: #2E7D32;
    }
    .pred-box {
        background-color: #E3F2FD;
        border-left: 5px solid #2196F3;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 1.15rem;
        font-weight: 600;
        color: #1565C0;
    }
    .webcam-card {
        background-color: #F3F4F6;
        border: 2px dashed #9CA3AF;
        border-radius: 10px;
        padding: 24px;
        text-align: center;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Cache model loading to avoid memory leaks
@st.cache_resource
def get_model():
    return load_model()

def convert_mpg_to_mp4(mpg_path: Path, output_mp4: Path) -> bool:
    """Converts MPG video to H.264 MP4 using imageio-ffmpeg bundled binary."""
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", str(mpg_path),
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",
            str(output_mp4)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return res.returncode == 0 and output_mp4.exists()
    except Exception as e:
        st.warning(f"Video conversion error: {e}")
        return False

def generate_lip_gif(frames_tensor: tf.Tensor, output_gif_path: Path):
    """Generates an animated GIF of the extracted lip region."""
    arr = frames_tensor.numpy()
    min_v = np.min(arr)
    max_v = np.max(arr)
    if max_v > min_v:
        uint8_frames = (((arr - min_v) / (max_v - min_v)) * 255.0).astype(np.uint8)
    else:
        uint8_frames = (arr * 255.0).astype(np.uint8)
    
    # Squeeze channel dim: (75, 46, 140)
    uint8_frames = np.squeeze(uint8_frames, axis=-1)
    imageio.mimsave(str(output_gif_path), uint8_frames, fps=15)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=70)
    st.title("Silent Speech AI")
    st.markdown("**Lip-Reading Computer Vision Model**")
    st.markdown("Predicts spoken sentences from silent video sequences using a 3D-CNN + Bidirectional LSTM network.")
    st.divider()

    st.subheader("Mode Selection")
    app_mode = st.radio("Choose Input Mode:", ["📁 Pre-Recorded Dataset", "📹 Real-Time Live Webcam"])
    st.divider()

    st.subheader("Model Specs")
    st.markdown("""
    - **Architecture:** 3D CNN + 2x BiLSTM
    - **Input Shape:** `(75, 46, 140, 1)`
    - **Vocabulary:** 41 character classes
    - **Dataset:** GRID Corpus (Speaker 1)
    """)
    st.divider()

    if app_mode == "📁 Pre-Recorded Dataset":
        st.subheader("Select Input Video")
        if VIDEOS_DIR.exists():
            video_options = sorted([f.name for f in VIDEOS_DIR.glob("*.mpg")])
        else:
            video_options = []

        if not video_options:
            st.error(f"No video files found in `{VIDEOS_DIR}`.")
            st.stop()

        selected_video_name = st.selectbox("Choose a sample video:", video_options, index=0)
        selected_stem = Path(selected_video_name).stem

# Main Layout
st.markdown('<div class="main-header">👄 Silent Speech Lip-Reading Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deep Learning-based visual speech recognition without audio signals</div>', unsafe_allow_html=True)

if app_mode == "📁 Pre-Recorded Dataset":
    selected_mpg_path = VIDEOS_DIR / selected_video_name
    selected_align_path = ALIGNMENTS_DIR / f"{selected_stem}.align"

    # Cache converted video in temp folder to avoid repeated conversions
    temp_dir = Path(tempfile.gettempdir()) / "lipreading_cache"
    temp_dir.mkdir(parents=True, exist_ok=True)
    cached_mp4_path = temp_dir / f"{selected_stem}.mp4"
    cached_gif_path = temp_dir / f"{selected_stem}_lips.gif"

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📹 Input Video")
        
        # Convert MPG to MP4 if needed
        if not cached_mp4_path.exists():
            with st.spinner("Converting MPG video to MP4 for browser playback..."):
                success = convert_mpg_to_mp4(selected_mpg_path, cached_mp4_path)
        else:
            success = True

        if success and cached_mp4_path.exists():
            with open(cached_mp4_path, "rb") as vf:
                st.video(vf.read())
        else:
            st.info("Video player preview unavailable; processing frames directly from MPG.")

        # Ground truth alignment
        ground_truth = load_alignment_text(selected_align_path)
        if ground_truth:
            st.markdown("**Ground Truth Utterance (Actual Spoken Words):**")
            st.markdown(f'<div class="gt-box">💬 "{ground_truth}"</div>', unsafe_allow_html=True)

    with col2:
        st.subheader("🧠 Deep Learning Inference")

        # Load and process frames
        with st.spinner("Extracting lip region & running 3D-CNN + BiLSTM..."):
            frames, alignments = load_data(selected_mpg_path)

            # Generate actual lip GIF if not cached
            if not cached_gif_path.exists():
                generate_lip_gif(frames, cached_gif_path)

            st.caption("Extracted Lip Region ROI (75 frames, 46x140 px grayscale):")
            if cached_gif_path.exists():
                st.image(str(cached_gif_path), width=320)

            # Load cached model
            model = get_model()

            # Run inference: shape must be (1, 75, 46, 140, 1)
            model_input = tf.expand_dims(frames, axis=0)
            yhat = model.predict(model_input, verbose=0)

            # CTC greedy decode
            ctc_decoded = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()

            # Convert to text filtering out -1 padding
            predicted_text = decode_prediction(ctc_decoded)

        # Display prediction
        st.markdown("**Model Predicted Sentence:**")
        st.markdown(f'<div class="pred-box">🎯 "{predicted_text}"</div>', unsafe_allow_html=True)

        if ground_truth:
            if predicted_text.strip().lower() == ground_truth.strip().lower():
                st.success("✅ **Exact Match!** The model recognized the spoken words with 100% sentence accuracy.")
            else:
                st.info("🔍 **Prediction Comparison:** Model transcript generated using CTC greedy decoding.")

        with st.expander("🔬 View Technical Details & Raw CTC Tokens"):
            st.write(f"**Input Tensor Shape:** `{model_input.shape}`")
            st.write(f"**Raw Output Probability Shape:** `{yhat.shape}`")
            flat_tokens = [int(x) for x in np.array(ctc_decoded).flatten() if x != -1]
            st.write("**Decoded Token IDs:**", flat_tokens)
            st.write(f"**Selected Video File:** `{selected_video_name}`")

else:
    # Real-Time Live Webcam Mode
    st.subheader("📹 Real-Time Live Camera Lip-Reading")
    st.markdown("""
    The real-time camera mode opens your live webcam feed, automatically tracks your face, isolates the **lip region (ROI)** with high-precision smoothing, and feeds consecutive frames directly into the LipNet neural network.
    """)

    col_cam1, col_cam2 = st.columns([1.2, 1], gap="large")

    with col_cam1:
        st.markdown("""
        ### 🚀 How to Launch the Live Camera
        You can launch the live camera lip-reader using the button below or by running `.\\cam` in your terminal.
        """)

        if st.button("🔴 Launch Real-Time Desktop Camera Lip-Reader", use_container_width=True, type="primary"):
            st.info("Starting live camera window... Look at your camera and press **SPACE** to speak!")
            python_exe = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
            live_cam_script = PROJECT_ROOT / "live_cam.py"
            subprocess.Popen([str(python_exe), str(live_cam_script)])

        st.markdown("""
        ```powershell
        # Or launch directly from terminal:
        .\\cam
        ```
        """)

        st.markdown("""
        ---
        #### 🎮 In-Camera Controls:
        - **`[SPACE BAR]`**: Starts recording **75 lip frames** (approx 2.5 seconds). Speak your sentence during the recording bar!
        - **`[C]`**: Toggles **Continuous Mode** (streams sliding window predictions).
        - **`[Q]` or `[ESC]`**: Exits and closes the camera feed.
        """)

    with col_cam2:
        st.markdown("### 💡 Tips for Best Lip-Reading Accuracy")
        st.markdown("""
        1. **Lighting:** Ensure your face is evenly lit with no heavy shadows over the mouth.
        2. **Positioning:** Sit straight facing the camera at a distance of about 0.5 to 1 meter.
        3. **Target Vocabulary (GRID Grammar):** The model is trained on GRID sentence patterns:
           - `<command>`: *bin, lay, place, set*
           - `<color>`: *blue, green, red, white*
           - `<preposition>`: *at, by, in, with*
           - `<letter>`: *a-z (e.g. f, p, s, etc.)*
           - `<digit>`: *zero-nine (e.g. two, five)*
           - `<adverb>`: *again, now, please, soon*
           - **Example:** *"bin blue at f two now"* or *"place green by a four please"*
        """)
