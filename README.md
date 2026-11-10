# SCT_ML_3: Cats vs Dogs Classification (Support Vector Machine)

Task 3 of the SkillCraft Technology Machine Learning internship.

**Goal:** classify images as cat or dog using a Support Vector Machine.

**Dataset:** Dogs vs. Cats. Images used: [number of images], split 80% train / 20% test. Each image is converted to grayscale, resized to 64×64, and described with HOG (histogram of oriented gradients) features.

## Method
1. Load images and label them from their folder names (cats or dogs).
2. Extract HOG features.
3. Standardise the features and train an RBF-kernel `SVC` (`C=1.0`, `gamma="scale"`).
4. Evaluate with accuracy, precision, recall and a confusion matrix.

## Results
| Metric | Value |
|---|---|
| Test accuracy | [value] |
| Cat precision / recall | [value] / [value] |
| Dog precision / recall | [value] / [value] |

`outputs/confusion_matrix.png` shows where the model confuses the two classes.

## How to run
1. Open `SCT_ML_3.ipynb` in Google Colab.
2. Upload the dataset zip and run the unzip and sorting cells in the notebook.
3. Run all cells (Runtime → Run all).

## Files
- `SCT_ML_3.ipynb`: Colab notebook with the full run and outputs
- `svm_cats_dogs.py`: the same code as a script
- `outputs/`: metrics, confusion matrix and saved model
