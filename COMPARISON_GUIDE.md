# 📊 Lip Reading AI — Dataset, Models & Training Analysis

> **Quick Reference Guide for Video Presentation & Code Review**

---

## 📁 1. GRID vs LRS3 Dataset Folder Structures

### A. GRID Corpus (Local in your project)
* **Location on disk:** [`data/`](file:///c:/Users/agarw/OneDrive/Desktop/Slient-Speech-Recognition-Automatic-Lip-reading-Model/data/)
```text
data/
├── s1/                              <-- 1,000 video files (.mpg)
│   ├── bbaf2n.mpg                   <-- 75 frames, 25 FPS (3 seconds)
│   ├── bbaf3s.mpg
│   └── ... (1,000 total)
└── alignments/
    └── s1/                          <-- 1,000 timestamp ground truth files (.align)
        ├── bbaf2n.align             <-- "bin blue at f two now"
        ├── bbaf3s.align
        └── ... (1,000 total)
```

### B. LRS3-TED Dataset (Oxford Research Dataset)
* **Source:** 5,000+ TED & TEDx talk videos (440+ hours)
* **Local Pretrained Model:** [`models/open_vsr/`](file:///c:/Users/agarw/OneDrive/Desktop/Slient-Speech-Recognition-Automatic-Lip-reading-Model/models/open_vsr/)
```text
LRS3-TED Dataset Structure (Original Research):
├── trainval/                        <-- ~30,000 video segments
├── test/                            <-- Benchmark test utterances
└── models/open_vsr/ (Your Project):
    ├── lipread_ctc.int8.onnx        <-- Trained weights (203 MB, INT8 quantized)
    └── tokens.json                  <-- 5,049 subword vocabulary tokens
```

---

## 🎯 2. LipNet vs Auto-AVSR Accuracy Comparison

| Metric / Feature | 🟡 LipNet (GRID Model) | 🟢 Auto-AVSR (LRS3 Model) |
| :--- | :--- | :--- |
| **Model Architecture** | 3D-CNN + 2× Bidirectional LSTM | ResNet-18 (Frontend) + Conformer (Self-Attention) |
| **Vocabulary Size** | **34 - 52 words** (Fixed grammar) | **5,049 subword tokens** (Open English) |
| **Input Crop Size** | 46 × 140 pixels | 88 × 88 pixels |
| **Word Error Rate (WER)** | **~4.8%** (on fixed GRID grammar) | **~25% - 28%** (Open English conversational sentences) |
| **Sentence Word Accuracy** | **~95.2%** | **~72% - 75%** |
| **Comparison to Humans** | Only knows 52 robotic words | **Beats or matches human lip-readers (30% - 60% accuracy)** |
| **Language Flexibility** | ❌ Fails on any normal word | ✅ Understands names, phrases & everyday speech |

---

## ⚖️ 3. Overfitting vs Generalization Comparison

| Analysis Dimension | ⚠️ LipNet (GRID Corpus) | ✅ Auto-AVSR (LRS3-TED) |
| :--- | :--- | :--- |
| **Overfitting Status** | **High Overfitting Risk** | **Well Regularized / High Generalization** |
| **Dataset Diversity** | Single speaker (`s1`), studio lighting, 52 fixed words | 5,000+ diverse speakers, various angles, real-world TED stages |
| **Grammar Memorization** | Memorizes fixed pattern: `[cmd] [color] [prep] [letter] [digit] [adv]` | Learns real phonetic & visual subword language structure |
| **Failure Mode** | If you say *"Hello"* or *"How are you"*, it outputs nonsense like *"bin blue"* | If you say *"Hello"*, it predicts *"HELLO"* |
| **Underfitting Status** | No underfitting (Model has 7M parameters, plenty for 52 words) | No underfitting (18-layer ResNet + Multi-Head Conformer has high capacity) |
| **Regularization Techniques**| Dropout (0.5) in BiLSTM layers | Massive dataset scale (440h), data augmentation (crop/flip), INT8 quantization |

---

## 🖼️ 4. Visual Comparison Charts in Your Folder

You also have pre-generated comparison diagrams inside [`images/`](file:///c:/Users/agarw/OneDrive/Desktop/Slient-Speech-Recognition-Automatic-Lip-reading-Model/images/):
1. **[`images/performance_comparison.png`](file:///c:/Users/agarw/OneDrive/Desktop/Slient-Speech-Recognition-Automatic-Lip-reading-Model/images/performance_comparison.png)** — Performance and accuracy benchmark charts.
2. **[`images/model_architecture.png`](file:///c:/Users/agarw/OneDrive/Desktop/Slient-Speech-Recognition-Automatic-Lip-reading-Model/images/model_architecture.png)** — Visual diagram of layer connections.
3. **[`images/feature_comparison.png`](file:///c:/Users/agarw/OneDrive/Desktop/Slient-Speech-Recognition-Automatic-Lip-reading-Model/images/feature_comparison.png)** — Feature differences table chart.
4. **[`images/processing_pipeline.png`](file:///c:/Users/agarw/OneDrive/Desktop/Slient-Speech-Recognition-Automatic-Lip-reading-Model/images/processing_pipeline.png)** — Full image pipeline diagram.

