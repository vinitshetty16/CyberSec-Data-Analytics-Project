# CyberSec-Data-Analytics-Project
This project focuses on constructing multi-class machine learning-based classification models for anomaly detection in network traffic. The goal is to identify various network traffic types by leveraging two ideal datasets for anomaly detection.

## Processes Involved
- **Data Sourcing**: Identify suitable datasets for anomaly detection in network traffic.
- **Data Filtering**: Remove anomalous data points from the datasets.
- **Data Analysis and Quality Assessment**: Analyze the datasets and assess their quality.
- **Feature Engineering**: Extract relevant features from the datasets for model training.
- **Predictive Modeling**: Train classification models using machine learning algorithms.
- **Data Visualization**: Visualize the data and model performance for better understanding.
- **Findings Distribution**: Share insights and findings derived from the models.

## Algorithms Utilized
1. **Random Forest**: Ensemble approach utilizing decision trees.
2. **Decision Tree (CART)**: Tree-structured classifier prone to overfitting.
3. **Naive Bayes**: Probabilistic classifier based on Bayes Theorem.
4. **K-Nearest Neighbour (KNN)**: Instance-based learning algorithm.
5. **Adaboost Classifier**: Iterative ensemble algorithm combining multiple classifiers.
6. **Logistic Regression**: Estimating likelihood of categorical dependent variables.
7. **Support Vector Machines (SVM)**: Hyperplane-based classification approach.
8. **Linear Discriminant Analysis (LDA)**: Dimensionality reduction technique.

## Evaluation Metrics
- **Accuracy**: Overall correctness of the model.
- **Precision**: Proportion of true positive among all positive predictions.
- **Recall**: Proportion of true positive identified correctly.
- **F1-Score**: Harmonic mean of precision and recall.
- **False Alarm (False Positive Rate)**: Proportion of false alarms among all negatives.

## Experiment Protocol
- Dataset 1: Divided attack classes into five categories.
- Dataset 2: Separated attack classes into two labels.
- Employed supervised algorithms and standardized numeric characteristics.
- Utilized hyperparameter tuning strategies like Halving RandomSearchCV and RandomizedSearchCV.

## Results
- Random Forest achieved the highest accuracy for Dataset 1 (76%) with balanced precision and recall.
- SVM surpassed all other classifiers with an accuracy of 77% for Dataset 2.

## Conclusion
- SVM, Random Forest, and KNN performed well across multiple metrics.
- Different algorithms showed varying performance on different datasets.
- Optimizing hyperparameters significantly improved model performance.
- Evaluation metrics provide insights into the strengths and weaknesses of each algorithm.

This README.md provides an overview of the project, detailing the datasets used, algorithms employed, evaluation metrics, experiment protocol, and results obtained. For a more detailed analysis, refer to the project documentation and codebase.
