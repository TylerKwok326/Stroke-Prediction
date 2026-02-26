# Stroke-Prediction

### Summary
This project focuses on building a classification model for stroke prediction using demographic, medical, and lifestyle data. The goal is to classify whether an individual is likely to experience a stroke (binary classification: stroke vs. no stroke) and demonstrate a complete supervised machine learning workflow. The dataset used in this project was obtained from Kaggle and consists of two files: a training dataset and a test dataset. The target variable, stroke, contains two classes indicating whether a stroke occurred or not.
Exploratory data analysis was conducted to understand feature distributions and their relationships with the target variable. By visualizing each feature against stroke vs. non-stroke cases, the analysis revealed a strong class imbalance, where the majority of samples belonged to the non-stroke class.

### Data Preprocessing
During initial exploration, the data was examined to understand feature types, distributions, and overall quality. Numerical features included age, glucose level, BMI, hypertension, and heart disease, while categorical features included gender, smoking status, work type, marital status, and residence type. Age values appeared in decimal format, so they were rounded and converted to integers. Rows containing the rare “Other” category in gender were removed. Duplicate rows and missing values were also checked before preprocessing.
During preprocessing, the ID column was removed since it does not contribute to prediction. Because the dataset contains both categorical and numerical features, categorical variables were encoded using two approaches: ordinal encoding for smoking status to preserve category ordering, and one-hot encoding for nominal features such as gender and work type. A ColumnTransformer was used to apply these transformations consistently while preserving numerical features.
The dataset was then split into training and validation sets using train_test_split. Since stroke cases were significantly fewer than non-stroke cases, class imbalance was addressed using SMOTE (Synthetic Minority Oversampling Technique) to generate synthetic samples of the minority class. This helped prevent models from becoming biased toward predicting the majority class.

### Modeling Approach
Several machine learning models were trained and evaluated as baseline classifiers, including Logistic Regression, Random Forest, XGBoost, LightGBM, and Voting Classifiers (hard and soft voting). Models were trained using the processed feature set and evaluated on a validation split. Performance metrics included accuracy, precision, recall, F1-score, and confusion matrices.
Because the dataset is imbalanced, the project emphasized recall rather than accuracy alone. In medical prediction tasks, missing true stroke cases is more critical than generating false positives. SMOTE was applied particularly with Logistic Regression to improve detection of minority cases, and model performance was compared across metrics to select an appropriate balance between precision and recall.

### Training Results
Model comparison showed that some models achieved high overall accuracy but relatively low recall, highlighting the importance of selecting evaluation metrics carefully for imbalanced datasets. Logistic Regression combined with SMOTE demonstrated improved recall, making it more suitable for identifying stroke cases even at the cost of slightly lower accuracy. Confusion matrices were used to visualize true positives, true negatives, false positives, and false negatives for each model, providing deeper insight into model behavior beyond single numerical metrics.


Takeaway: By raw accuracy, Logistic Regression performed the best , however this is misleading due to the class imbalance only identifying only 3 out of 134 positive cases. Applying SMOTE to Logistic Regression significantly reduced false negatives from 131 to 31, but increase in false positives

