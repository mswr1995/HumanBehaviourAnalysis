# Human Behaviour Analysis

A video action recognition project using the UCF101 dataset to classify human actions in videos. This repository contains two different implementations for comparison.

## Overview

This project implements deep learning approaches for recognizing human actions in videos using the UCF101 dataset. The system extracts frames from videos, uses pre-trained CNN models for feature extraction, and trains neural networks to classify actions across 101 different categories.

## Dataset

The project uses the [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php), which contains:
- 13,320 videos from 101 action categories
- Actions include sports, musical instruments, human-object interaction, and body-motion activities
- Standard train/test splits provided by the dataset

## Implementations

### 1. CNN-RNN Architecture (Jupyter Notebook)
**File**: `195112046_Human_Behavior_Analysis.ipynb`

A more sophisticated approach using temporal modeling:
- **Feature Extraction**: Pre-trained VGG16 CNN for spatial features
- **Temporal Modeling**: LSTM layers to capture temporal dependencies between frames
- **Architecture**: CNN-RNN hybrid with high-dimensional LSTM layers (456 → 8 units)
- **Input**: Sequences of video frames with masking for variable lengths
- **Approach**: Considers both spatial and temporal information in videos

### 2. Simple CNN + FC Architecture (Python Scripts)
**Files**: `Model.py`, `Train.py`, `Evaluate.py`, `VideoPreprocessing.py`

A simpler frame-based classification approach:
- **Feature Extraction**: Pre-trained VGG16 for feature extraction from individual frames
- **Classification**: Fully connected neural network with dropout layers
- **Architecture**: 5-layer FC network (1024→512→256→128→101 neurons)
- **Input**: Individual frames (224x224x3)
- **Approach**: Frame-level classification without explicit temporal modeling

## Project Structure

```
├── 195112046_Human_Behavior_Analysis.ipynb  # CNN-RNN implementation
├── VideoPreprocessing.py                    # Video frame extraction utilities
├── Model.py                                 # Simple CNN+FC model architecture
├── Train.py                                 # Training pipeline for simple model
├── Evaluate.py                              # Model evaluation and prediction
├── UCF/                                     # Dataset directory
│   ├── ucfTrainTestlist/                    # Train/test split files
│   ├── train_1/                             # Extracted frames
│   └── ckpt/                                # Model checkpoints
└── README.md
```

## Usage

### CNN-RNN Implementation (Recommended)
Open and run the Jupyter notebook:
```bash
jupyter notebook 195112046_Human_Behavior_Analysis.ipynb
```

### Simple CNN+FC Implementation
1. Place UCF101 videos in the appropriate directory
2. Run the training pipeline:
```python
python Train.py
```
3. Evaluate the trained model:
```python
python Evaluate.py
```

## Requirements

- Python 3.x
- TensorFlow/Keras 2.5+
- OpenCV
- NumPy
- Pandas
- scikit-learn
- tqdm
- Jupyter Notebook (for CNN-RNN implementation)

## Key Differences

| Aspect | CNN-RNN (Notebook) | CNN+FC (Scripts) |
|--------|-------------------|------------------|
| Temporal modeling | ✅ LSTM layers | ❌ Frame-independent |
| Complexity | Higher | Lower |
| Performance | Better (considers temporal info) | Good (simpler baseline) |
| Training time | Longer | Faster |
| Memory usage | Higher | Lower |

## Notes

- The CNN-RNN approach is more sophisticated and typically performs better for video classification
- The simple CNN+FC approach serves as a good baseline and is easier to understand
- Both implementations use VGG16 pre-trained on ImageNet for feature extraction
- Frame extraction is done at 1 FPS for computational efficiency
- Model checkpoints are saved during training for both approaches

This project demonstrates different approaches to video action recognition, from simple frame-based classification to more advanced temporal modeling techniques.
