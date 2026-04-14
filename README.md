# AR Image Overlay on Chessboard

This project estimates the camera pose from a chessboard video and overlays an image onto the chessboard plane.

---

## Overview

The program detects a chessboard pattern in each frame of a video, estimates the camera pose using the detected corners, and projects a 2D image onto the chessboard surface.

---

## How It Works

1. Read video frames from input file  
2. Detect chessboard corners using OpenCV  
3. Refine corner locations with `cornerSubPix`  
4. Estimate camera pose using `solvePnP`  
5. Define a rectangular plane on the chessboard  
6. Project the plane onto the image using `projectPoints`  
7. Warp and overlay the image onto the projected region  

---

## AR Object

- A transparent PNG image is used as the AR object  
- The image is projected onto the chessboard plane using perspective transformation  

---

## Input

- Video file (chessboard video)
- Camera parameters (JSON file)
  - camera matrix
  - distortion coefficients
- Overlay image (PNG with transparency)

---

## Output

- Video with the image projected onto the chessboard  

---
