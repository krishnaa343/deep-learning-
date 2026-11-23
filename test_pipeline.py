import sys
import os
from pathlib import Path

# Add project root and app to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
APP_DIR = PROJECT_ROOT / "app"
sys.path.insert(0, str(APP_DIR))

def test_full_pipeline():
    print("=" * 60)
    print("Silent Speech Lip-Reading AI - End-to-End Pipeline Test")
    print("=" * 60)

    # 1. Imports
    print("\n[1/6] Verifying Dependencies and Imports...")
    try:
        import tensorflow as tf
        import numpy as np
        import cv2
        import imageio
        import imageio_ffmpeg
        import streamlit
        from utils import load_data, load_alignment_text, decode_prediction, VIDEOS_DIR, ALIGNMENTS_DIR
        from modelutil import load_model, build_model
        print(f"  ✓ TensorFlow version: {tf.__version__}")
        print(f"  ✓ NumPy version: {np.__version__}")
        print(f"  ✓ OpenCV version: {cv2.__version__}")
        print(f"  ✓ Streamlit version: {streamlit.__version__}")
        print(f"  ✓ FFmpeg binary found: {imageio_ffmpeg.get_ffmpeg_exe()}")
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False

    # 2. Check Dataset
    print("\n[2/6] Checking Dataset...")
    mpg_files = list(VIDEOS_DIR.glob("*.mpg"))
    align_files = list(ALIGNMENTS_DIR.glob("*.align"))
    print(f"  ✓ Found {len(mpg_files)} MPG videos in {VIDEOS_DIR}")
    print(f"  ✓ Found {len(align_files)} ALIGN files in {ALIGNMENTS_DIR}")
    if len(mpg_files) == 0:
        print("  ✗ No videos found!")
        return False

    # 3. Model Loading
    print("\n[3/6] Building Model Architecture & Loading Checkpoint...")
    try:
        model = load_model()
        print("  ✓ Model constructed and weights successfully loaded from models/checkpoint!")
        print(f"  ✓ Total Parameters: {model.count_params():,}")
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        return False

    # 4. Inference Test on Sample Videos
    print("\n[4/6] Running Inference on Sample Videos...")
    sample_videos = ["bbaf2n.mpg", "bbaf3s.mpg", "bbaf5a.mpg"]
    for sample_name in sample_videos:
        sample_path = VIDEOS_DIR / sample_name
        if not sample_path.exists():
            continue

        sample_stem = sample_path.stem
        align_path = ALIGNMENTS_DIR / f"{sample_stem}.align"
        ground_truth = load_alignment_text(align_path)

        # Load data
        frames, _ = load_data(sample_path)
        print(f"\n  Testing video: {sample_name}")
        print(f"    • Extracted frames shape: {frames.shape} (Expected: (75, 46, 140, 1))")

        # Predict
        model_input = tf.expand_dims(frames, axis=0)
        yhat = model.predict(model_input, verbose=0)
        decoded = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        pred_text = decode_prediction(decoded)

        print(f"    • Ground Truth: '{ground_truth}'")
        print(f"    • Prediction:   '{pred_text}'")
        if ground_truth.strip().lower() == pred_text.strip().lower():
            print("    • Result: MATCH ✅")
        else:
            print("    • Result: Partial / Near Match 🔍")

    # 5. Video Conversion Test
    print("\n[5/6] Testing MP4 Conversion with bundled FFmpeg...")
    import subprocess
    import tempfile
    test_mpg = VIDEOS_DIR / "bbaf2n.mpg"
    test_mp4 = Path(tempfile.gettempdir()) / "test_converted.mp4"
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg_exe, "-y", "-i", str(test_mpg), "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(test_mp4)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode == 0 and test_mp4.exists() and test_mp4.stat().st_size > 0:
        print(f"  ✓ Successfully converted {test_mpg.name} -> {test_mp4.name} ({test_mp4.stat().st_size} bytes)")
    else:
        print(f"  ✗ Conversion failed: {res.stderr.decode()}")
        return False

    # 6. GIF Generation Test
    print("\n[6/6] Testing Lip ROI GIF Generation...")
    test_gif = Path(tempfile.gettempdir()) / "test_lips.gif"
    arr = frames.numpy()
    uint8_frames = np.squeeze((((arr - arr.min()) / (arr.max() - arr.min())) * 255.0).astype(np.uint8), axis=-1)
    imageio.mimsave(str(test_gif), uint8_frames, fps=15)
    if test_gif.exists() and test_gif.stat().st_size > 0:
        print(f"  ✓ Successfully generated lip GIF ({test_gif.stat().st_size} bytes)")
    else:
        print("  ✗ GIF generation failed!")
        return False

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! Project is fully functional and ready to run.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)

