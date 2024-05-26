# Capstone-Project-Final-Year

# Fraud Transaction Detection using Blockchain and Machine Learning

**Author:** Aditya Prakash,Amit Tiwari, Bhuwan Chauhan   
**Email:** adityaprakash6986@gmail.com

## Abstract
Recent research has shown that machine learning (ML) techniques have been applied very effectively to the problem of payments-related fraud detection. Such ML-based techniques have the potential to evolve and detect previously unseen patterns of fraud. In this project, we apply multiple ML techniques based on Logistic Regression and Support Vector Machine (SVM) to the problem of payments fraud detection using a labeled dataset containing payment transactions. We demonstrate that our proposed approaches can detect fraudulent transactions with high accuracy and a reasonably low number of false positives.

## Introduction
Digital payment systems are rapidly being adopted worldwide. Credit card and payment companies are experiencing rapid growth in transaction volume. In the third quarter of 2018, PayPal Inc. processed 143 billion USD in total payment volume. Alongside this transformation, there has also been a rapid increase in financial fraud within these payment systems.

An effective fraud detection system should detect fraudulent transactions with high accuracy and efficiency, ensuring genuine users are not prevented from accessing the payments system. A large number of false positives can lead to poor customer experience and loss of customers.

A major challenge in applying ML to fraud detection is the presence of highly imbalanced datasets. In many available datasets, the majority of transactions are genuine, with an extremely small percentage of fraudulent ones. Designing an accurate and efficient fraud detection system that minimizes false positives while effectively detecting fraudulent activity is a significant challenge for researchers.

In this project, we apply multiple binary classification approaches—Logistic Regression, Linear SVM, and SVM with RBF kernel—on a labeled dataset that consists of payment transactions. Our goal is to build binary classifiers that can separate fraudulent transactions from non-fraudulent ones. We compare the effectiveness of these approaches in detecting fraud.

## Relevant Research
Several ML and non-ML based approaches have been applied to the problem of payments fraud detection. The paper [1] reviews and compares multiple state-of-the-art techniques, datasets, and evaluation criteria applied to this problem, discussing both supervised and unsupervised ML-based approaches involving ANN (Artificial Neural Networks), SVM (Support Vector Machines), HMM (Hidden Markov Models), clustering, etc. Paper [5] proposes a rule-based technique for fraud detection. Paper [3] discusses the problem of imbalanced data that results in a high number of false positives and proposes techniques to alleviate this problem. Paper [2] proposes an SVM-based technique to detect metamorphic malware, discussing the problem of imbalanced datasets and how to successfully detect them with high precision and accuracy.

## Dataset and Analysis
In this project, we use a Kaggle-provided dataset [8] of simulated mobile-based payment transactions. We analyze this data by categorizing it with respect to different types of transactions. We also perform PCA (Principal Component Analysis) to visualize the variability of data in a two-dimensional space. The dataset contains five categories of transactions: 'CASH IN', 'CASH OUT', 'DEBIT', 'TRANSFER', and 'PAYMENT'. The details are provided in Table I below.

| Transaction Type | Non-fraud Transactions | Fraud Transactions | Total |
|------------------|------------------------|--------------------|-------|
| CASH IN          | 1,399,284              | 0                  | 1,399,284 |
| CASH OUT         | 2,233,384              | 4,116              | 2,237,500 |
| TRANSFER         | 528,812                | 4,097              | 532,909 |
| DEBIT            | 41,432                 | 0                  | 41,432 |
| PAYMENT          | 2,151,494              | 0                  | 2,151,494 |
| **TOTAL**        | 6,354,407              | 8,213              | 6,362,620 |

The dataset consists of around 6 million transactions, out of which 8,213 transactions are labeled as fraud, resulting in a highly imbalanced dataset with only 0.13% fraud transactions. We display the result of performing two-dimensional PCA on subsets for two transaction types that contain frauds—TRANSFER and CASH OUT transactions.

### PCA Decomposition
The PCA decomposition of TRANSFER transactions shows high variability across two principal components for non-fraud and fraud transactions, indicating that the TRANSFER dataset can be linearly separable. Our chosen algorithms, Logistic Regression and Linear SVM, are likely to perform well on such a dataset, as shown in Figure 1 below.

![TRANSFER transactions](link-to-image) ![CASH OUT transactions](link-to-image)

## Method
Our goal is to separate fraud and non-fraud transactions by obtaining a decision boundary in the feature space defined by input transactions. Each transaction can be represented as a vector of its feature values. We have built binary classifiers using Logistic Regression, Linear SVM, and SVM with RBF kernels for TRANSFER and CASH OUT sets, respectively.

### A. Logistic Regression
Logistic regression is used to find a linear decision boundary for a binary classifier. For a given input feature vector \( x \), a logistic regression model with parameter \( \theta \) classifies the input \( x \) using the hypothesis \( h_{\theta}(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}} \), where \( g \) is the Sigmoid function. The logistic loss function with respect to parameters \( \theta \) is given by:

\[ J(\theta) = \frac{1}{m} \sum_{i=1}^m \log \left( 1 + \exp(-y^{(i)} \theta^T x^{(i)}) \right) \]

### B. Support Vector Machine
Support Vector Machine (SVM) creates a classification hyperplane in the space defined by input feature vectors. The optimization problem can be characterized by:

\[ \min_{\gamma, w, b} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^m \xi_i \]
\[ \text{subject to } y^{(i)} (w^T x^{(i)} + b) \geq 1 - \xi_i, \quad \xi_i \geq 0, \quad i = 1, \ldots, m \]

We use two variants of SVM: Linear SVM and SVM based on RBF kernel. The RBF kernel function on two vectors \( x \) and \( z \) in the input space is defined as:

\[ K(x, z) = \exp \left( -\frac{\|x - z\|^2}{2\sigma^2} \right) \]

### C. Class Weights Based Approach
We assign different weights to samples belonging to fraud and non-fraud classes to counter the data imbalance problem. We penalize mistakes made in misclassifying fraud samples more than non-fraud samples. We fine-tune our models by choosing class weights to obtain an optimal balance between precision and recall scores on our fraud class samples. This design trade-off enables us to balance detecting fraud transactions with high accuracy and preventing a large number of false positives.

## Experiments
We describe our dataset split strategy and the training, validation, and testing processes. All software was developed using the Scikit-learn [7] ML library.

### A. Dataset Split Strategy
We divided our dataset based on different transaction types. We use TRANSFER and CASH OUT transactions for our experiments. For both types, we divided respective datasets into three splits: 70% for training, 15% for cross-validation (CV), and 15% for testing. We use stratified sampling to maintain the same proportion of each class in a split as in the original dataset.

### B. Model Training and Tuning
We trained our models using a class weight-based approach, evaluating their performance on the CV split. We chose class weights that provided the highest recall on the fraud class with no more than ~1% false positives. Finally, we used the models to make predictions on our test dataset split.

## Results and Discussion
We evaluated the performance of our models using metrics such as recall, precision, f1-score, and area under the precision-recall curve (AUPRC).

### A. Class Weights Selection
We observed that higher class weights resulted in higher recall at the cost of lower precision on our CV split. For the TRANSFER dataset, the effect of increasing weights is less prominent, especially for Logistic Regression and Linear SVM algorithms.

### B. Results on Train and Test Sets
We obtained high recall and AUPRC scores for TRANSFER transactions, with ~0.99 recall for all three algorithms. For CASH OUT transactions, the results were less promising. However, all proposed approaches detected fraud transactions with high accuracy and low false positives, especially for TRANSFER transactions.

## Conclusion and Future Work
In fraud detection, we often deal with highly imbalanced datasets. For the Paysim dataset, our proposed approaches detect fraud transactions with high accuracy and low false positives, particularly for TRANSFER transactions. Fraud detection involves a trade-off between correctly detecting fraudulent samples and minimizing false positives. 

We can further improve our techniques by using algorithms like Decision Trees to leverage categorical features and by interpreting the Paysim dataset as a time series to build time series-based models using algorithms like CNN. Creating user-specific models based on previous transactional behavior can also enhance our decision-making process.
