# Problem Set 01: Pediatric Chest X-ray Classification

## Approach

This problem set explores a pediatric chest X-ray dataset and builds a binary CNN classifier for the `NORMAL` and `PNEUMONIA` classes. The raw dataset remains outside this submission directory and is never modified.

## Methodology

1. Inspect the `train`, `val`, and `test` directory structure.
2. Count images by split and class.
3. Inspect image dimensions and file metadata.
4. Visualize representative images and class distributions.
5. Resize images to `150 x 150` grayscale inputs and normalize pixels to `[0, 1]`.
6. Apply random flipping, rotation, and zoom augmentation only to training images.
7. Define a three-block Sequential CNN with pooling, dropout, and a sigmoid output.
8. Train with Adam at learning rate `0.0001`, `EarlyStopping(restore_best_weights=True)`, and `ReduceLROnPlateau`.
9. Evaluate the untouched test split using a classification report, confusion matrix, ROC curve, and ROC-AUC.

## Findings

The stabilized model produced the following recorded test-set results at a probability threshold of `0.5`:

- Test accuracy: `0.9054`
- NORMAL recall: `0.8248`
- PNEUMONIA recall: `0.9538`
- ROC-AUC: `0.9613`

PNEUMONIA recall is especially important because a false negative represents a real pneumonia case classified as normal. In a screening context, minimizing missed pneumonia cases is more important than relying on overall accuracy alone.

## Files

- `Problem_Set_01_Dataset_Exploration.ipynb`: complete exploration, preprocessing, CNN training, and evaluation workflow.
- `src/data_loader.py`: reusable TensorFlow dataset-loading and preprocessing helper.

## Running the notebook

Run the notebook from the repository root with the project virtual environment selected:

```powershell
.venv\Scripts\python.exe
```

The notebook expects the dataset at `data/chest_xray`. The dataset is intentionally not included in the Git submission.
