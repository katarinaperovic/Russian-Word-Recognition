# Russian Word Recognition using OCR

## Overview

This project focuses on recognizing Russian words from images using Optical Character Recognition (OCR) and minimizing the total distance between the recognized and actual words. The dataset is located in the `data1/pictures` folder, and the correct words for each image are provided in `data1/res.csv`. The objective is to achieve the lowest total word distance across all images.

---

## Project Structure

```
📂 Russian-Word-Recognition
 ├── 📂 data1                 # Dataset folder
 │   ├── 📂 pictures          # Images containing Russian words
 │   ├── 📝 res.csv           # Correct words corresponding to each image
 ├── 📜 SC23-G3-RA-186-2020.py    # Main Python script for OCR & evaluation
 ├── 📜 README.md             # Project documentation
```

---

## Requirements

To run this project, ensure you have the following dependencies installed:

```bash
pip install numpy opencv-python pytesseract pandas
```

---

## How to Run

1. Open a terminal and navigate to the project directory:
   ```bash
   cd Russian-Word-Recognition
   ```
2. Run the script to process images:
   ```bash
   python SC23-G3-RA-186-2020.py
   ```

---

## Implementation Details

- The program reads images from `data1/pictures` and applies **OCR (Tesseract)** for word recognition.
- **Preprocessing techniques** include:
  - Grayscale conversion
  - Adaptive thresholding
  - Noise removal using morphological operations
- The recognized words are compared with `res.csv` to calculate the total distance using:
  - **Hamming Distance** for words of the same length
  - **Substring Hamming Distance + Length Difference** for words of different lengths

---

## Distance Calculation

- If the actual and recognized words have the same length, the **Hamming Distance** is used:
  ```
  distance("pera", "pera") = 0
  distance("Pera", "pero") = 2
  ```
- If they have different lengths, the shortest substring Hamming distance is used, plus the absolute length difference:
  ```
  distance("zika", "zikac") = 1
  distance("zika", "zivac") = 2
  ```

---

## Results & Optimization

- The solution is optimized to achieve a **total distance ≤ 1**
- Various preprocessing and OCR tuning techniques were tested to improve accuracy.

---

## Future Improvements

- Implement **deep learning-based** OCR for improved accuracy.
- Optimize character segmentation for better word recognition.
- Enhance preprocessing to handle different lighting and font variations.

---

## Author

- **Katarina Perović**

---

## License

This project is for educational purposes only and follows the guidelines set for the Soft Computing 2023/24 course.

