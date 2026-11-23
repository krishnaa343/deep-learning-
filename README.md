# 👄 Silent Speech Recognition — Automatic Lip Reading AI

> **A real-time deep learning system that recognizes spoken words solely by observing a person's lip movements, without using any audio.**

---

## 🎯 1. Problem Statement & Applications

### What is Silent Speech Recognition?
Our goal is to recognize spoken words **purely from visual lip movements**, eliminating the need for sound waves or acoustic signals.

### Real-World Applications:
- 🛡️ **Surveillance & Security**: Transcribing silent CCTV security camera footage.
- 🦻 **Assisting Hearing-Impaired Individuals**: Automatic visual transcription in everyday life.
- ✈️ **Noisy Environments**: High-noise zones like airports, factories, and traffic where microphones fail.
- 🤫 **Confidential & Silent Communication**: Secure, silent voice transmission in private or military settings.

---

## ⚙️ 2. Project Functionality & How It Works

Our complete pipeline processes live visual speech in **6 real-time steps**:

```
[1. Live Webcam Feed] (640 × 480)
         │
         ▼
[2. Face Detection] (OpenCV Haar Cascade Classifier)
         │
         ▼
[3. Lip ROI Extraction] (88 × 88 Grayscale Mouth Crops)
         │
         ▼
[4. Frame Buffering] (Motion Auto-Detect OR Manual [SPACE] key)
         │
         ▼
[5. Deep Learning Model] (Spatial-Temporal Features + CTC Greedy Decoding)
         │
         ▼
[6. Real-Time Subtitle Panel] (Displays recognized sentence below video feed)
```

1. **Webcam Capture**: Live camera stream captured at **640×480 resolution**.
2. **Face & Mouth Detection**: OpenCV's **Haar Cascade classifier** locates the face in each frame and calculates the mouth coordinates (located at ~77% of face height).
3. **Lip Preprocessing**: The mouth region is cropped, converted to **grayscale**, and resized to **88×88 pixels**.
4. **Recording Trigger**: 
   - **Automatic Mode**: Movement sensor starts recording when lips move and stops upon silence.
   - **Manual Mode**: Press **`[SPACE]`** to start speaking, and press **`[SPACE]`** again to transcribe.
5. **CTC Sequence Decoding**: Converts consecutive lip motion representations into English characters and words.
6. **On-Screen Subtitles**: Output text appears immediately in the bottom subtitle bar.

---

## 🧠 3. Dual Model Architecture

Our project implements two complementary deep learning architectures:

| Feature | 🟡 Model 1: LipNet | 🟢 Model 2: Auto-AVSR *(Demo Model)* |
| :--- | :--- | :--- |
| **Model Type** | 3D-CNN + Bidirectional RNN | ResNet-18 + Conformer Encoder |
| **Dataset** | **GRID Corpus** (Speaker 1) | **LRS3-TED** (Oxford University) |
| **Vocabulary** | 34 fixed words (Closed vocabulary) | **Open-Vocabulary English** (5,049 subwords) |
| **Architecture** | 3× Conv3D + 2× BiLSTM + CTC | 18-Layer ResNet + Multi-Head Self-Attention + CTC |
| **Input Shape** | `(75, 46, 140, 1)` | `(1, 1, T, 88, 88)` |
| **Accuracy** | ~95% on GRID fixed grammar | ~75% open-domain English (Competitive with humans) |
| **Primary Use** | Dataset verification & benchmark | **Live Webcam Open-Speech Recognition** |

> 🔴 **Note for Demonstration**: For our live webcam demonstration, we run **Model 2 (Auto-AVSR)** to recognize open-vocabulary English sentences on the spot.

---

## 🚀 4. How to Run

### Live Open-Vocabulary Camera (Model 2):
```powershell
.\run
```
*(Or double-click `run.bat` in File Explorer)*

- **`[SPACE]`**: Start recording / Stop & transcribe sentence
- **`[C]`**: Toggle continuous rolling prediction
- **`[Q]` / `[ESC]`**: Exit

---

## 📁 5. Project Directory Structure

```
Slient-Speech-Recognition-Automatic-Lip-reading-Model/
├── open_cam.py              # Real-time Auto-AVSR open-vocabulary webcam lip reader
├── run.bat                  # 1-click launcher for the live camera
├── app/
│   ├── streamlitapp.py      # Streamlit web application
│   ├── modelutil.py         # LipNet 3D-CNN + BiLSTM architecture
│   └── utils.py             # Data loading & CTC decoding utilities
├── data/
│   ├── s1/                  # 1,000 video files (.mpg) from GRID corpus
│   └── alignments/s1/       # 1,000 alignment ground-truth files (.align)
└── models/
    ├── checkpoint*          # LipNet weights (GRID Corpus)
    └── open_vsr/
        ├── lipread_ctc.int8.onnx  # Auto-AVSR quantized ONNX model (203 MB)
        └── tokens.json            # 5,049 open-vocabulary subword tokens
```
