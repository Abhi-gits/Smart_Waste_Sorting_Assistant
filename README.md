# Smart Waste Sorting Assistant


A notebook-based AI project that classifies waste images, generates disposal guidance, and recommends a bin using a simple reinforcement learning (RL) policy.

Source notebook: `Smart_Waste_Sorting_Assistant.ipynb`

Link to colab notebook: https://colab.research.google.com/drive/1g-b14FxwK0xVStRZn5rgUplfsY8HMb7l?usp=sharing


## Project Overview

This project combines three components:

1. **Computer Vision (CV)**: Classifies a waste image into one of 6 classes.
2. **NLP-style Explanation Layer**: Returns a human-readable disposal recommendation for the predicted class.
3. **Reinforcement Learning (RL) Bin Decision**: Chooses one of 3 bins using an RL agent trained in a custom Gymnasium environment.

The full pipeline is executed via:

- `smart_waste_assistant(image_path)`

## Features

- Transfer learning with **MobileNetV2** for image classification.
- Dataset loading from directory structure using `ImageDataGenerator`.
- Training/validation split (80/20).
- Accuracy visualization across epochs.
- Text explanations mapped per waste type.
- RL-based bin selection (`Recycling`, `Organic`, `Landfill`).

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Gymnasium
- Stable-Baselines3 (PPO)
- Pillow

## Dataset

The notebook expects a zip file:

- `trashnet-dataset.zip`

Expected extraction path:

- `/content/trashnet-dataset`

Class folders are expected under that directory (e.g., `metal`, `plastic`, etc.).

## Notebook Pipeline

### 1. Install Dependencies

Installs required libraries with pip.

### 2. Import Libraries

Loads TensorFlow, Keras modules, and utility libraries.

### 3. Prepare Dataset

- Extracts `trashnet-data.zip` into `/content`.
- Prints folder contents for quick verification.

### 4. Load Data

- Uses `ImageDataGenerator(rescale=1./255, validation_split=0.2)`.
- Creates training and validation generators:
  - Input image size: `224x224`
  - Batch size: `32`
  - Class mode: `categorical`

### 5. Build Classifier

- Base model: `MobileNetV2(weights="imagenet", include_top=False)`
- Head: `GlobalAveragePooling2D` + `Dense(6, softmax)`
- Compile:
  - Optimizer: `adam`
  - Loss: `categorical_crossentropy`
  - Metric: `accuracy`

### 6. Train Classifier

- `epochs=3` (fast demonstration training)

### 7. Plot Metrics

Plots train and validation accuracy from `history`.

### 8. Predict Waste Type

Function:

- `predict_waste(img_path)`

Process:

- Load image
- Resize to `224x224`
- Normalize to `[0,1]`
- Run model inference
- Return predicted class label

### 9. Generate Explanation

Function:

- `generate_explanation(waste_type)`

Uses a dictionary mapping each waste class to a disposal explanation string.

### 10. RL Environment + Agent

Environment class:

- `WasteEnv(gym.Env)`

Agent:

- PPO (`stable_baselines3.PPO`)
- Trained for `2000` timesteps.

Bin selector:

- `choose_bin()` returns one of:
  - `Recycling Bin`
  - `Organic Bin`
  - `Landfill Bin`

### 11. Full Assistant

Function:

- `smart_waste_assistant(image_path)`

Outputs:

- Predicted waste type
- Disposal explanation
- Recommended bin

## How to Run

### Option A: Google Colab (recommended for current paths)

1. Open the notebook in Colab.
2. Upload `trashnet-dataset.zip` to `/content`.
3. Run cells top-to-bottom.
4. Test with:
   - `smart_waste_assistant("/content/trashnet-dataset/metal/metal101.jpg")`

### Option B: Local Jupyter

1. Create a Python environment.
2. Install dependencies:

```bash
pip install tensorflow stable-baselines3 gymnasium matplotlib pillow
```

3. Update hardcoded `/content/...` paths to local paths.
4. Run the notebook sequentially.

## Current Limitations (Important)

- RL environment is highly simplified:
  - `correct_action` is fixed to `0` in `step()`, so the policy is not class-aware.
  - `choose_bin()` is not currently tied to the predicted waste class.
- Classifier training is short (`3` epochs), so accuracy may be limited.
- No model saving/loading is included.
- No robust evaluation metrics (confusion matrix, precision/recall, etc.) are included.
- Notebook includes shell-style commands (`!pip`, `!ls`) suited to notebook runtime.

## Suggested Improvements

1. Link bin decision to predicted class using deterministic mapping or state-aware RL reward design.
2. Freeze/unfreeze MobileNetV2 layers for better transfer learning control.
3. Train for more epochs with callbacks (`EarlyStopping`, `ModelCheckpoint`).
4. Add evaluation report (classification report + confusion matrix).
5. Save models (`model.save`) and load for inference-only usage.
6. Convert notebook into modular Python package/app (`data.py`, `model.py`, `inference.py`, `rl.py`, `main.py`).

## Example Output (from notebook)

```text
Predicted Waste Type: plastic

Disposal Explanation:
Plastic should be recycled because it can be reused to produce new materials.

Recommended Bin:
Recycling Bin
```


## License

No license file is currently included. Add a `LICENSE` file if you plan to share or publish this project.
