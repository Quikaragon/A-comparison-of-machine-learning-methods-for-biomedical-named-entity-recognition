# TFG: Biomedical Named Entity Recognition

This repository contains notebooks and scripts to compare machine learning methods for biomedical named entity recognition (NER) using the JNLPBA dataset.

## Requirements

- Python 3.10 (recommended: use conda)
- Git
- Anaconda/Miniconda

## Step-by-step Setup

### 1. Clone the repository
```bash
git clone https://github.com/Quikaragon/A-comparison-of-machine-learning-methods-for-biomedical-named-entity-recognition.git
cd A-comparison-of-machine-learning-methods-for-biomedical-named-entity-recognition
```

### 2. Create the conda environment
```bash
conda env create -f environment_TFG.yml
conda activate TFG
```

> **Note:** Make sure to download or have the `environment_TFG.yml` file in this folder before creating the conda environment. You can find it in the repository or request it from the project owner if missing.

### 3. Install Jupyter (if not included)
```bash
conda install jupyter
```

### 4. Open the notebook
```bash
jupyter notebook
```
Open `ExploringDataset.ipynb` or any other notebook in the project.

### 5. Select the correct kernel
In Jupyter, select the Python kernel corresponding to the `TFG` environment.

### 6. Download the dataset (automatic)
The notebooks automatically download the JNLPBA dataset using the `datasets` library.

### 7. Run the cells
Follow the cell order to reproduce the experiments and visualizations.

## Main Notebooks
- ExploringDataset.ipynb: Initial dataset exploration and analysis.
- CRF_Seqeval.ipynb: Basic CRF models.
- CRF_Seqeval-Extension.ipynb: CRF models with extended features.
- Bert.ipynb: Training and evaluation with BERT.
- BioBert.ipynb: Training and evaluation with BioBERT.

## Images and Results
Plots and results are saved in the `Imagenes/` folder.

## Reproducibility
If you have dependency issues, check the `environment_TFG.yml` file and make sure to install the environment from it.

## Notes on Performance and Hardware

- **CRF Hyperparameter Search:**
  - The hyperparameter search for CRF models (RandomizedSearchCV) can take a long time (from several minutes to over an hour), depending on your CPU and the number of iterations.
  - For faster results, reduce the number of iterations or use a machine with more CPU cores.

- **Transformers (BERT/BioBERT):**
  - Training and inference with transformer models is much faster and more practical with a GPU (CUDA-compatible, e.g., NVIDIA).
  - On CPU, training can take several hours or even days, while on GPU it is usually completed in minutes or a few hours.
  - It is highly recommended to use a machine with a GPU for running the BERT and BioBERT notebooks.

- If you encounter out-of-memory errors, try reducing the batch size or sequence length in the notebook configuration.

## Contact
For questions or suggestions, open an issue in the repository or contact the author.
