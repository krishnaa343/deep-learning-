import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D,
    Activation, TimeDistributed, Flatten
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoint"

def build_model() -> Sequential:
    """Constructs the LipNet 3D-CNN + BiLSTM architecture."""
    model = Sequential(name="LipNet")

    # 1st 3D Conv block
    model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    # 2nd 3D Conv block
    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    # 3rd 3D Conv block
    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    # Flatten spatial features across time
    model.add(TimeDistributed(Flatten()))

    # 1st Bidirectional LSTM layer
    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    # 2nd Bidirectional LSTM layer
    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    # Dense layer for 41 output character classes
    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))

    return model

def load_model(checkpoint_path: Path = CHECKPOINT_PATH) -> Sequential:
    """Builds the LipNet model and loads the trained checkpoint weights."""
    model = build_model()
    
    ckpt_str = str(checkpoint_path)
    if os.path.exists(ckpt_str + ".index") or os.path.exists(ckpt_str):
        model.load_weights(ckpt_str).expect_partial()
    else:
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_str}")

    return model
