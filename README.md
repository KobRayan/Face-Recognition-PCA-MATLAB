# Face Recognition using Eigenfaces (PCA)

## Project Overview
This project implements a facial recognition system using Principal Component Analysis (PCA) and Linear Algebra concepts in MATLAB. It analyzes a dataset of 90 training images and 30 test images to identify individuals and classify objects as "Face" or "Non-Face" based on reconstruction error.

## Technical Features
* **Algorithm:** Eigenfaces using Singular Value Decomposition (SVD).
* **Dimensionality Reduction:** Compressed 4096-pixel vectors (64x64 images) into essential features.
* **Classification:** Implemented a threshold-based classifier to distinguish between human faces and other objects.
* **Performance:** Achieved ~90% identification accuracy with k=30 principal components.

## Repository Structure
* **main.m**: Core script handling data loading, PCA training, and identification logic.
* **imageto64.m**: Helper function to reshape vector data into 64x64 matrices.
* **Rayan Kobrossly compte_rendu Matlab.pdf**: Detailed engineering report (in French) analyzing the results and math.

## Note on Data
This repository contains the source code. The dataset files (YaleFaces.mat and baseunknown.mat) are excluded for copyright and file size reasons. The code demonstrates the algorithmic logic structure.

---
Created as part of the Engineering Curriculum at Polytech Nancy.
