
# README.md مدقق Project

## Overview

This project is dedicated to advancing the field of Re-Identification (ReID) using deep learning. It encompasses a comprehensive suite of Python scripts for configuring project settings, creating and compiling models with modern neural network backbones, and defining custom dataset classes for efficient data handling. The project supports tasks that require computing cosine similarity between features, crucial for applications like image re-identification or verification systems.

## Features

- **Dataset Handling**: Two custom dataset classes (`ReID_Dataset1` and `ReID_Dataset`) cater to different dataset structures, ensuring balanced batches and uniform image input sizes.
- **Model Training**: Utilizes EfficientNet and ResNet backbones for model creation, incorporating a custom CosineSimilarity layer.


## Custom CosineSimilarity Layer and Functions

This script defines a custom CosineSimilarity layer, along with functions for calculating similarity targets, similarity loss, similarity recall, and similarity scores for a dataset. It is designed for use in models that require an understanding of the similarity between different inputs.

- **CosineSimilarity**: A keras layer that computes cosine similarity between samples in its input.
- **similarity_targets**: Generates a matrix indicating if pairs of samples are similar (1) or different (0), adjusted to avoid negative similarity values.
- **similarity_loss**: A custom loss function that calculates binary cross-entropy between true labels and predictions, with an option to weight positive samples more heavily.
- **similarity_recall**: Computes the recall for positive samples based on a similarity threshold, excluding samples marked with -1.
- **similarity_scores**: Evaluates a model on a dataset, returning similarity scores for "same" and "different" image pairs, facilitating model evaluation and tuning.


- **Project Configuration**: A configuration script (`config.py`) streamlines the setup, defining key aspects such as dataset paths, model weights, and save paths.
- **Library Management**: Centralized import of necessary libraries and modules through `libraries.py`, facilitating ease of use across different project components.

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/blackice2090/
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure the Project**:
   - Set your dataset path, dataset name, and other configurations in `config.py`.
4. **Prepare the Datasets**:
   - Use `prepare_datasets` to initialize and split the dataset into training and validation sets.
5. **Train Your Model**:
   - Execute the model script to create and compile your model.

## Usage

To utilize this project for your ReID tasks:

1. **Data Preparation**:
   - Adjust `dataset_path` and `REID_NAME` in `config.py` to match your dataset requirements.
2. **Model Selection**:
   - Choose between EfficientNet and ResNet presets in `config.py` for your model backbone.
3. **Training and Evaluation**:
   - Run `train.ipynb` for model training and `analysis.ipynb` for performance evaluation.

## Contributing

Contributions are welcome! Please read through our contributing guidelines for more details on how to submit pull requests and issues.

## Authors

- **Team Work** - Initial Work - [مدقق](https://github.com/blackice2090)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Special thanks to everyone who has contributed to the development and testing of this project.
- Inspired by the challenges in the Re-Identification field and the potential impact of deep learning solutions.
    