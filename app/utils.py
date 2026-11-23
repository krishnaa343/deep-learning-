import os
from pathlib import Path
from typing import List, Tuple, Union
import cv2
import numpy as np
import tensorflow as tf

# Define project directories dynamically relative to this file
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
VIDEOS_DIR = DATA_DIR / "s1"
ALIGNMENTS_DIR = DATA_DIR / "alignments" / "s1"

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# Mapping integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_video(path: Union[str, Path]) -> tf.Tensor:
    """Extracts, grayscales, crops, and normalizes frames from a video file."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frame = tf.image.rgb_to_grayscale(frame)
        # Crop lip region: height 190:236 (46px), width 80:220 (140px), channels 1
        cropped = frame[190:236, 80:220, :]
        frames.append(cropped)
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Could not read any frames from video: {path}")

    frames_tensor = tf.cast(frames, tf.float32)
    mean = tf.math.reduce_mean(frames_tensor)
    std = tf.math.reduce_std(frames_tensor)
    if std == 0:
        return frames_tensor - mean
    return (frames_tensor - mean) / std

def load_alignments(path: Union[str, Path]) -> tf.Tensor:
    """Loads token alignment from .align file."""
    with open(str(path), 'r', encoding='utf-8') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 3 and parts[2] != 'sil':
            tokens = [*tokens, ' ', parts[2]]
    
    if not tokens:
        return tf.constant([], dtype=tf.int64)

    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def load_alignment_text(path: Union[str, Path]) -> str:
    """Returns the plain text ground truth sentence from the alignment file for display."""
    if not os.path.exists(str(path)):
        return ""
    with open(str(path), 'r', encoding='utf-8') as f:
        lines = f.readlines()
    words = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 3 and parts[2] != 'sil':
            words.append(parts[2])
    return " ".join(words)

def load_data(path: Union[str, Path, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Loads video frames and alignment labels.
    Preserves 4D tensor shape (75, 46, 140, 1) without squeezing.
    """
    if isinstance(path, tf.Tensor):
        path_str = bytes.decode(path.numpy())
    else:
        path_str = str(path)

    file_stem = Path(path_str).stem
    video_path = VIDEOS_DIR / f"{file_stem}.mpg"
    alignment_path = ALIGNMENTS_DIR / f"{file_stem}.align"

    if not video_path.exists():
        if Path(path_str).exists():
            video_path = Path(path_str)

    frames = load_video(str(video_path))
    alignments = load_alignments(str(alignment_path)) if alignment_path.exists() else tf.constant([], dtype=tf.int64)
    
    # Preserves (75, 46, 140, 1) shape so Conv3D receives (1, 75, 46, 140, 1)
    return frames, alignments

def decode_prediction(decoder_output) -> str:
    """
    Converts CTC decoded indices into readable text,
    properly filtering out blank and padding tokens (-1).
    """
    if decoder_output is None:
        return ""
    flat_tokens = np.array(decoder_output).flatten()
    valid_tokens = [int(t) for t in flat_tokens if t != -1]
    if not valid_tokens:
        return ""
    char_tensors = num_to_char(valid_tokens)
    return tf.strings.reduce_join(char_tensors).numpy().decode('utf-8')
