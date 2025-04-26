# Miniprojekt 2. Semester

## Project Description
This project automatically computes the score of a **Kingdomino** board based on an image of the assembled tiles.
It combines **image processing**, **feature extraction**, **machine learning (LDA + KNN)**, and **custom crown detection** to predict terrain types, detect crowns, and calculate the final board points.

Key features:
- Extracts features like color histograms and texture information from board images.
- Uses **Linear Discriminant Analysis (LDA)** for feature reduction.
- Classifies each tile using **K-Nearest Neighbors (KNN)**.
- Detects crowns separately with a custom-built crown detection pipeline.
- Computes full board points using clustering based on terrain labels and crowns.

---

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/miniprojekt_2s.git
    ```

2. Navigate to the project directory:
    ```bash
    cd miniprojekt_2s
    ```

3. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage
1. Ensure your datasets are placed correctly:
   - Cropped board images should be under:
     ```
     data/images/
     ```
   - Label file (CSV) should be located at:
     ```
     data/labels.csv
     ```
   - Crown template image should be:
     ```
     data/crown_template.png
     ```

2. Run the main script:
    ```bash
    python main.py
    ```

---

## How It Works
- **Dataset Creation**:
  - Images are split into 5x5 grids.
  - Features are extracted for each tile: HSV color histograms and texture features.
  - Labels are read from CSV files.
- **Feature Scaling**:
  - Features are standardized with `StandardScaler`.
- **Dimensionality Reduction**:
  - `LinearDiscriminantAnalysis (LDA)` is used to reduce feature space.
- **Classification**:
  - A `KNeighborsClassifier` predicts the terrain type for each tile.
- **Crown Detection**:
  - Crowns are detected independently using template matching in both HSV and grayscale modes.
- **Point Calculation**:
  - The board is scored based on connected regions of the same terrain, weighted by crown counts.

---

## Evaluation
The project evaluates model performance based on:
- **Tile classification accuracy** (terrain labels).
- **Crown detection accuracy**.
- **Board points exact match accuracy**.
- **Mean Absolute Error (MAE)** and **R² score** for predicted vs. true board points.

All metrics are reported after the test phase.

---

## Folder Structure
```plaintext
data/
├── images/                 # Cropped board images
├── crown_template.png      # Crown template
├── labels.csv              # Tile labels
miniprojekt_2s/
├── main.py                 # Main pipeline
├── preprocessing.py        # Image preprocessing
├── crown_class.py          # Crown detection class
├── visualizer.py           # Visualization tools
├── requirements.txt        # libraries required
```

---

## Quick Notes
- Tested with Python 3.10+.
- Ensure OpenCV (`opencv-python`) is installed.
- Paths are adapted for macOS/Linux style (`/`). Windows users might need small path adjustments.

