# ML_Project
## Description
More than 500 million people worldwide suffer from type 2 diabetes and even more live at an increased risk of developing the disease. Type 2 diabetes is the most common type of diabetes and can lead to heart diseases, strokes and other severe conditions. Therefore, an early diagnosis and risk assessment is key for the prevention of serious damage. Nevertheless, people with this type of diabetes often live many years without being diagnosed although there exist specific factors that could indicate a high risk of a patient. Machine learning techniques can help predicting type 2 diabetes based on given risk factors and hence lead to earlier and better treatment.

Following, the aim of this work is to investigate the performance of different machine learning models for the prediction of type 2 diabetes based on given risk factors. Since missing a diagnosis of type 2 diabetes in early stages can have life-threatening consequences, our secondary goal will be to minimize the number of false negatives in our models.
## Installation
```bash
conda create --name my_env
conda activate my_env
conda install pip
pip install -r requirements.txt
```
Set the conda environment on jupyter notebook
```bash
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=my_env
```

## Instructions
Please refer to the notebooks listed below. Each notebook shows the final result as well as intermediate results.
### 1. Preprocessing
[EDA_and_Preprocessing.ipynb](EDA_and_Preprocessing.ipynb)
### 2. Models
- [Logistic Regression](Logistic%20Regression.ipynb)
- [SVM](SVM.ipynb)
- [Decision Tree and Random Forest](Tree%20Classifier%20-%20Random%20Forest.ipynb)
- [XGBoost](XGBoost.ipynb)
- [Neural Network](neural_network.ipynb)
### 3. Finetuning
[Fine_tuning.ipynb](Fine_tuning.ipynb)
### 4. Evaluation
[Final evaluation.ipynb](Final%20evaluation.ipynb)

## Other important files
### Dataset
- [Raw dataset](diabetes_dataset.arff)
- [Unbalanced dataset](diabetes_dataset_preprocessed.csv) and [oversampled dataset](diabetes_dataset_preprocessed_oversampled.csv) after preprocessing
### Models saved after hyperparameter search
We saved our models after grid and randomized search. They can be found as `.pkl` files in `data` directory. 
### Related materials
Please refer to `sources` directory for our sources of evaluation as well as our project proposal
