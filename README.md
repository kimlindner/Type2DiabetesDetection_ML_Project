# ML_Project
## Description
More than 500 million people worldwide suffer from type 2 diabetes and even more live at an increased risk of developing the disease. Type 2 diabetes is the most common type of diabetes and can lead to heart diseases, strokes and other severe conditions. Therefore, an early diagnosis and risk assessment is key for the prevention of serious damage. Nevertheless, people with this type of diabetes often live many years without being diagnosed although there exist specific factors that could indicate a high risk of a patient. Machine learning techniques can help predicting type 2 diabetes based on given risk factors and hence lead to earlier and better treatment.

Following, the aim of this work is to investigate the performance of different machine learning models for the prediction of type 2 diabetes based on given risk factors. Since missing a diagnosis of type 2 diabetes in early stages can have life-threatening consequences, our secondary goal will be to minimize the number of false negatives in our models.
### Installation
```bash
python3 -m venv venv
source venv/bin/activate

pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
```

We have created a mini test set (100 samples) for QuickDraw dataset. Refer to [generating_100_test_data.ipynb](generating_100_test_data.ipynb) for further detail on how we extract these 100 samples.

We have created a special file [final_test.py](final_test.py) for this mini test set. You can evaluate all of our models on the mini test set using this file. For training, evaluating, or transfer learning on different datasets and different settings, please refer to [main.py](main.py) and [core.py](core.py).

## Instructions
Please refer to the notebooks listed below. Simply run each notebook, final result as well as intermediate results will be found.
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

