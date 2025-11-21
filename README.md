# 📡 CSI Sensing Platform (ESP32 + Processing Toolkit)

This repository provides a complete **CSI (Channel State Information) sensing platform** using ESP32. It includes everything from firmware for data collection to processing, visualization, feature extraction, and machine learning pipelines.

The goal is to build a full Wi-Fi sensing workflow for research and real-world applications.

---

## 🧩 1. System Overview

**CSI Pipeline:**

```
ESP32 → CSI Collection → UART/TCP → Data Logging →
Preprocessing → Feature Extraction → ML/DL → Applications
```

This project includes:

* **ESP32 CSI firmware**
* **Python/C++ processing toolkit**
* **Realtime visualization tools**
* **Dataset management structure**
* **Examples and utilities**

Applications include:

* Human Activity Recognition (HAR)
* Device-free localization
* Gesture recognition
* Breathing & fall detection
* General Wi-Fi sensing

---

## 📁 2. Repository Structure

```
csi-sensing/
│
├── firmware/                     # ESP32 CSI firmware
├── data/                         # CSI datasets (raw / processed / labels)
├── processing/                   # Processing algorithms (Python/C++)
├── examples/                     # Demo scripts
├── tools/                        # Utility scripts for capture/analysis
├── docs/                         # Technical documentation
├── configs/                      # Configuration files
├── scripts/                      # Bash automation scripts
└── README.md
```

---

## 🔧 3. Building ESP32 Firmware

### Requirements

* ESP-IDF ≥ 5.0
* Python 3.8+
* Git & ESP32 toolchain installed

### Build

```bash
cd firmware/esp32-csi-collector
idf.py set-target esp32
idf.py build
idf.py -p /dev/ttyUSB0 flash
idf.py monitor
```

---

## 📥 4. Collecting CSI Data

### Capture via UART

```bash
python tools/logger.py \
    --port /dev/ttyUSB0 \
    --baud 921600 \
    --output data/raw/csi.log
```

### Capture via TCP/WebSocket

```bash
python tools/realtime_server.py --tcp 3333 --save data/raw/
```

---

## 🧹 5. CSI Preprocessing

```bash
python processing/python/preprocess.py \
    --input data/raw/csi.log \
    --output data/processed/csi_clean.npy
```

Includes:

* Noise removal
* Phase unwrapping
* Amplitude normalization
* Spike removal / smoothing
* Filtering (Hampel, Butterworth, SavGol)

---

## 📊 6. Visualization Tools

### Plot CSI over time

```bash
python examples/python-live-plot/plot_from_csi_serial.py -p <PORT>
```

### Heatmap visualization

Supported for:

* Subcarrier amplitude
* Subcarrier phase
* Time progression

---

## 🤖 7. Machine Learning / Deep Learning

Feature extraction:

```bash
python processing/python/feature_extract.py \
    --input data/processed/csi_clean.npy \
    --output data/processed/features.npy
```

Train SVM:

```bash
python processing/python/model/svm_classifier.py
```

Train LSTM:

```bash
python processing/python/model/lstm_model.py --epochs 40
```

---

## ⚡ 8. Realtime Demo

Plot CSI in realtime:

```bash
python examples/python-live-plot/live_plot.py --tcp 3333
```

---

## 📘 9. Documentation

Located in `docs/`:

* Project architecture
* CSI frame structure
* ESP32 CSI extraction guide
* Preprocessing techniques
* Sensing application notes

---

## 🤝 10. Contributing

Contributions are welcome!

* Open issues for bugs/feature requests
* Submit pull requests
* Discussion is encouraged

---

## 📜 11. License

MIT License.

