# MPEG-4 Video Encoder Pipeline — Implementation & Testing Report

## 1. Overview

This document outlines the implementation and testing procedures for the simplified MPEG-4 Video Encoder and Decoder pipeline. The pipeline processes sequential frames, encodes them using intra-frame (I-frames) and inter-frame (P-frames) prediction techniques natively used in MPEG standards, and packages them into a compressed binary bitstream.

## 2. Implementation Architecture

The project structure is broken down into modular pipeline stages.

### 2.1 Pre-processing (`pipeline/preprocessing.py`)

- **Color Space Conversion**: Uses OpenCV to convert input BGR images into YCbCr (Luminance and Chrominance) color space.
- **Chroma Subsampling**: Applies `4:2:0` chroma subsampling by downsampling the Cb and Cr channels by a factor of 2 in both dimensions using Area interpolation. This reduces the data size while preserving visual quality, as human eyes are more sensitive to luminance (Y) than chrominance.

### 2.2 Intra-Frame Coding (`pipeline/intra_coding.py`)

Used for independent **I-frames**.

- **Block Splitting**: The Y, Cb, and Cr channels are processed in 8x8 blocks, padded with zeros if necessary.
- **Transform**: Applies the 2D Discrete Cosine Transform (DCT) to convert spatial blocks into the frequency domain.
- **Quantization**: High-frequency components are discarded by dividing the DCT coefficients with a scaled standard JPEG Luminance Quantisation Matrix (controlled by a factor `Q`), followed by integer rounding.
- **Decoder Side**: Dequantizes by multiplying with the matrix, then applies IDCT to reconstruct the channels.

### 2.3 Inter-Frame Coding (`pipeline/inter_coding.py`)

Used for **P-frames** relying on motion estimation.

- **Block Matching**: Partitions the current frame's Y channel into 16x16 macroblocks. It searches the **reconstructed** reference frame inside a given `search_range` to find the highest correlating block using Sum of Absolute Differences (SAD).
- **Motion Vectors**: Records the `(dx, dy)` motion vector for each macroblock that yields the lowest SAD.
- **Residual Coding**: Subtracts the predicted reference block from the current block to produce a residual map. The residual is then DCT-transformed and quantised just like an I-frame, natively supporting negative integers.

### 2.4 Entropy Coding (`pipeline/entropy_coding.py`)

- **Serialization**: Packages the structured metadata, frame types, I-frame coefficients, P-frame residuals, and motion vectors into a dictionary payload.
- **Compression**: Transforms the dictionary into a binary byte string using Python's `pickle` and compress it using lossless DEFLATE via `zlib.compress(level=9)`.

### 2.5 Integration (`encoder.py`, `decoder.py`)

- `encoder.py` manages a Group of Pictures (GOP). For example, `GOP=8` forces one I-frame to be followed by 7 P-frames. Reference frames are actively decoded during encoding to prevent drift between the encoder and decoder.
- `decoder.py` handles parsing the bitstream and executing the decoding sequence linearly.

---

## 3. Testing Guide

### 3.1 Setup

Ensure all dependencies are satisfied:

```bash
pip install -r requirements.txt
```

Prepare a directory named `frames/` containing sequentially named input images (e.g. `frame_01.png`, `frame_02.png`). For rapid testing, use short, low-resolution clips (e.g., a few frames at 160x120 pixels).

### 3.2 Encoding Test

Run the encoder, adjusting the Quantization (`--Q`) and Group of Pictures (`--GOP`) flags.

```bash
python encoder.py --frames frames/ --output output/video.bin --GOP 4 --Q 1.0
```

**Verification Check:**

- The console will log: `Encoded frame X as I-frame/P-frame`.
- `output/video.bin` should successfully generate without crashing.

### 3.3 Decoding Test

Run the decoder to decompress the frames:

```bash
python decoder.py --input output/video.bin --output_dir output/frames/
```

**Verification Check:**

- The console will log: `Decoded I-frame/P-frame to output/frames/frame_000X.png`.
- Check the `output/frames/` directory to see the newly reconstructed sequence.
- Visually compare the output against the original instances. Artifacts might be subtly visible depending on your `Q` factor, but the images should be well preserved.

### 3.4 Evaluation (`evaluate.py`)

You can use `evaluate.py` to mathematically verify the integrity.

- **PSNR (> 30 dB)**: A measure of the error. Highly compressed states should still retain a ~30 dB PSNR, guaranteeing high visual fidelity. You can use the `compute_metrics` function inside the file to calculate this.
- **Compression Ratio**: Comparing the uncompressed frame byte size with `video.bin` proves the mathematical compression worked by heavily reducing the file size.
- **Pipeline Visualisation**: Add logic to independently run `visualize_pipeline("frames/", "output/plot.png")` in `evaluate.py` to render a 5-row plot displaying your Raw Image -> Channels -> DCT/Quantization flow -> Motion Vectors -> Decoder Residual layout.
