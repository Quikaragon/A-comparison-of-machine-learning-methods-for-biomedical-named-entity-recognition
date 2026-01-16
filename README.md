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

## Model Checkpoint Files (.pt)

After training the BERT or BioBERT models, a file named `bert_jnlpba_best.pt` or `biobert_jnlpba_best.pt` is generated in the project directory. These files contain the learned weights of the trained models and are required for making predictions or resuming training later.

- **File names:** `bert_jnlpba_best.pt`, `biobert_jnlpba_best.pt`
- **Approximate size:** 420 MB each
- **Purpose:** Store the parameters of the best model found during training (based on F1 score on the test set).
- **Usage:**
    - To use a trained model for inference or further training, load the file with `model.load_state_dict(torch.load("bert_jnlpba_best.pt"))` (or the corresponding BioBERT file).
    - Make sure the file is present in your working directory when running inference or evaluation code.

> **Note:** These files are not included in the repository due to their size. You will need to train the models yourself to generate them.

## Reproducibility

## Alternative: Using requirements.txt if the YAML environment fails

If you have trouble creating the environment with the YAML file, you can use the requirements.txt file as an alternative. Follow these steps:

1. **Unzip the project files** if you downloaded them as a ZIP.
2. **Create a new environment**:
  ```
  conda create -n TFG python=3.10 pip
  ```
3. **Activate the environment**:
  ```
  conda activate TFG
  ```
4. **Install the dependencies** (including CUDA support for PyTorch):
  ```
  pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
  ```
5. **Open the notebooks** with Jupyter Notebook or a similar tool:
  ```
  jupyter notebook
  ```

This method will install all required packages and allow you to run the notebooks even if the YAML environment creation fails.

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
For questions or suggestions, open an issue in the repository .
