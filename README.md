# 🤖 AI Hand Gesture Projects

> Control your computer with just your hand — AI-powered gesture recognition projects built with Python, OpenCV, and MediaPipe.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.31-orange?style=flat)
![License](https://img.shields.io/badge/License-MIT-brightgreen?style=flat)

---

## 📁 Project Structure

```
Py_AI_Model/
├── 🖱️  AI_Virtual_Mouse/
│   └── Ai_virtual_mouse.py     # Control mouse cursor using hand gestures
├── 🖐️  Hand_Tracker/
│   └── Handtracker.py          # Real-time hand landmark detection
├── main.py                     # Entry point
├── util.py                     # Utility functions
├── requirements.txt            # Dependencies
└── .gitignore
```

---

## 🚀 Projects

### 🖱️ AI Virtual Mouse
Control your computer's mouse cursor using just your hand in front of a webcam — no physical mouse needed.

**How it works:**
- Detects hand landmarks in real-time via webcam
- Index finger controls cursor position on screen
- Pinch gesture triggers mouse click
- Freeze cursor mode to prevent jitter

**Tech used:** `OpenCV` · `MediaPipe` · `PyAutoGUI`

---

### 🖐️ Hand Tracker
A real-time hand tracking module that detects and draws hand landmarks using your webcam feed.

**How it works:**
- Captures live webcam frames
- Processes frames through MediaPipe Hands
- Draws all 21 hand landmarks and connections on screen

**Tech used:** `OpenCV` · `MediaPipe`

---

## ⚙️ Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/Abhinikesh/Py_AI_Model.git
cd Py_AI_Model
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run

**AI Virtual Mouse:**
```bash
cd AI_Virtual_Mouse
python Ai_virtual_mouse.py
```

**Hand Tracker:**
```bash
cd Hand_Tracker
python Handtracker.py
```

> Make sure your webcam is connected and accessible. Press `q` to quit.

---

## 📦 Requirements

```
opencv-python
mediapipe==0.10.31
pyautogui
```

---

## 👤 Author

**Abhinikesh Kumar**
- GitHub: [@Abhinikesh](https://github.com/Abhinikesh)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">⭐ Star this repo if you found it cool!</p>
