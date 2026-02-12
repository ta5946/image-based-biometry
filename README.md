# Image Based Biometry (IBB)

This repository contains materials and assignments for the Image Based Biometry (IBB) course. The project is organized into four main assignments, each focusing on different biometric modalities and techniques.

## Project Structure

The project is divided into the following main directories:

### [Assignment 1](./Assignment1/) - Fingerprint Recognition
Focuses on fingerprint biometric systems, including minutiae extraction and matching.
- **Key Files:**
  - `fingerprint_recognition.ipynb`: Jupyter notebook for fingerprint analysis.
  - `fingerprint_recognition.py`: Python implementation of fingerprint recognition.
  - `IBBReport_A1.pdf`: Assignment report.
  - `DB2_B/`, `DB3_B/`, `DB4_B/`: Fingerprint databases.

### [Assignment 2](./Assignment2/) - Iris Recognition
Focuses on iris biometric systems.
- **Key Files:**
  - `hdbif.py`: Main script for iris recognition tasks.
  - `modules/irisRecognition.py`: Core iris recognition logic.
  - `eer.py`: Equal Error Rate calculation.
  - `IBBLabs_A2.pdf`: Assignment instructions.

### [Assignment 3](./Assignment3/) - Face Recognition (Classical)
Focuses on classical face recognition techniques such as HOG, LBP, and SIFT.
- **Key Files:**
  - `src/face_recognition.ipynb`: Jupyter notebook for face recognition experiments.
  - `IBBReport_A3.pdf`: Assignment report.
  - `plot/`: Visualization of CMC curves for different descriptors.

### [Assignment 4](./Assignment4/) - Deep Face Recognition
Focuses on deep learning-based face recognition using models like FaceNet and InsightFace.
- **Key Files:**
  - `src/deep_face_recognition.ipynb`: Jupyter notebook for deep face recognition.
  - `IBBReport_A4.pdf`: Assignment report.
  - `plot/`: Visualization of CMC curves for deep models.

## Tracked Files Overview

The repository tracks source code (.py, .ipynb), documentation (.pdf, .md), and necessary datasets/plots.

- **Source Code:** Located primarily in assignment-specific `src/` or root assignment folders.
- **Documentation:** Reports and instructions are available as PDFs within each assignment folder.
- **Data:** Datasets for fingerprints and face images are included in the tracked files for reproducibility.

## Requirements

The project uses Python and common data science/computer vision libraries:
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow/PyTorch (for deep learning assignments)
- Jupyter Notebook
