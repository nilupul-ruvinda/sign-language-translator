# Real-Time Sign Language Translator (A–Z)

A real-time Sign Language fingerspelling translator built from scratch in Python. Using a standard webcam, the application tracks 21 3D hand landmarks via **MediaPipe** and classifies static hand gestures (A–Z) using a custom **TensorFlow/Keras** neural network to construct words on screen in real time.

## Key Features
* **Custom Dataset Collector:** Interactive OpenCV capture tool with live skeleton overlay to record wrist-normalized hand coordinates.
* **Lightweight Landmark Model:** Trains on 63 normalized $(x, y, z)$ coordinates per hand rather than raw pixels, enabling fast CPU training and low-latency real-time inference.
* **Live Word Builder:** Includes frame-hold debouncing to prevent duplicate characters, plus keyboard controls (`Space` for spaces, `C` to clear).

## Tech Stack
* **Computer Vision & Tracking:** OpenCV, MediaPipe Hands
* **Machine Learning:** TensorFlow / Keras, NumPy, Scikit-Learn

## Project Structure
* `01_collect_letters.py` — Captures and normalizes 100 landmark samples per letter (A–Z).
* `02_train_letters.py` — Trains the dense neural network and exports `letter_model.h5`.
* `03_realtime_letters.py` — Runs live webcam inference to translate signs into text.

##user instructions:

Hold a sign steady for 1-2 seconds to type that letter into the Word bar at the top.

Spacebar: Add a space between words.

C: Clear the typed text.

Q: Exit the translator.

bakespace: clear the previous letter.
