# SCT_ML_3 – Cats vs Dogs Classification (Support Vector Machine)

**Task:** Implement a support vector machine (SVM) to classify images of cats and dogs from the Kaggle dataset.
**Dataset:** Dog/Cats Dataset (link in the internship brief).

## Setup
```bash
pip install -r requirements.txt
```
Unzip the dataset into `data/` so that cat and dog images sit in folders whose names start with `cat` or `dog`, for example `data/cats/` and `data/dogs/`. Sub-folders such as `train/` and `test/` are fine.

## Run
```bash
python svm_cats_dogs.py --data data --max-per-class 1000
python svm_cats_dogs.py --predict path/to/some_image.jpg
```

## Method
1. Collect images for both classes and sample up to `--max-per-class` from each.
2. Convert to grayscale, resize to 64×64 and extract HOG (histogram of oriented gradients) features.
3. Standardise the features and train an RBF-kernel `SVC` (80/20 stratified split).
4. Report accuracy, precision, recall and the confusion matrix.

## Outputs (`outputs/`)
- `metrics.txt`: accuracy and classification report
- `confusion_matrix.png`
- `svm_cats_dogs.joblib`: trained model

## Results
Test accuracy: ____
