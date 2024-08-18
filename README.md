
# Sports Articles Classification

This repository contains two Jupyter Notebooks that demonstrate different approaches to classifying sports articles as either "objective" or "subjective" using machine learning models.

## Notebooks Overview

### 1. Deep Learning Model for Sports Articles Classification

This notebook uses a deep learning approach to classify sports articles based on various text features.

- **Data Preprocessing**:
  - Loaded the sports articles dataset and explored its structure.
  - Split the dataset into features (`X`) and labels (`y`).
  - Scaled the feature data using `StandardScaler` and encoded the labels using `LabelEncoder`.

- **Model Development**:
  - Built a Sequential neural network with multiple dense layers using TensorFlow and Keras.
  - Compiled the model with binary cross-entropy loss and the Adam optimizer.
  - Trained the model over 100 epochs.

- **Evaluation**:
  - Evaluated the model's accuracy on the test dataset.
  - Generated predictions and a classification report.

- **Model Persistence**:
  - Saved the trained model to a file.
  - Demonstrated how to load the model and evaluate its performance.

- **Results**:
  - Accuracy: 82.8%
  - Loss: 0.4313
  - The classification report showed strong precision, recall, and F1-scores for both classes.

### 2. Machine Learning Models for Sports Articles Classification

This notebook explores multiple traditional machine learning algorithms to classify sports articles.

- **Data Preprocessing**:
  - Loaded the sports articles dataset and performed similar preprocessing steps as in the deep learning notebook, including feature scaling and label encoding.

- **Model Development and Evaluation**:
  - **Logistic Regression**:
    - Accuracy: Training - 86%, Testing - 84%
  - **Support Vector Machine (SVM)**:
    - Accuracy: Training - 88.4%, Testing - 84.8%
  - **K-Nearest Neighbors (KNN)**:
    - Accuracy: Training - 83.87%, Testing - 84.4%
  - **Decision Tree**:
    - Accuracy: Training - 100%, Testing - 74.4%
  - **Random Forest**:
    - Accuracy: Training - 100%, Testing - 83.6%
  - **Gradient Boosting**:
    - Accuracy: Training - 96.93%, Testing - 83.6%
  - **AdaBoost**:
    - Accuracy: Training - 89.33%, Testing - 83.6%

- **Results**:
  - The Support Vector Machine, Random Forest, and Gradient Boosting models provided the best balance between training and testing accuracy.
  - Precision, recall, and F1-scores were calculated for each model, with the SVM model showing strong overall performance.

## Installation

To run these notebooks, ensure you have Python installed with the following libraries:

```bash
pip install pandas scikit-learn tensorflow
```

## How to Use

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repository/sports-articles-classification.git
   cd sports-articles-classification
   ```

2. **Open the Notebooks**:
   - Use Jupyter Notebook or any compatible IDE to open and run the notebooks.

3. **Run the Cells**:
   - Execute each cell sequentially to preprocess the data, train the models, and evaluate their performance.

## Results Summary

- The deep learning model achieved a final accuracy of 82.8% on the test data, demonstrating the potential of neural networks in text classification tasks.
- Traditional machine learning models such as SVM and Gradient Boosting also performed well, with the SVM model achieving an accuracy of 84.8%.

## Conclusion

These notebooks provide a comprehensive approach to text classification using both deep learning and traditional machine learning methods. The results indicate that while deep learning models offer flexibility and strong performance, traditional models like SVM and Gradient Boosting can be equally effective with proper tuning.

## Acknowledgments

Thanks to the contributors of TensorFlow, scikit-learn, and the broader data science community for their invaluable resources and tools that made this project possible.
