# PD prediction
 PD prediction is a machine learning model used to diagnose Parkinson's disease. The model distinguishes samples from the three cohorts of Control, PD, and Prodromal by plasma metabolome data.
 
# Model

SVM-RFE (Support Vector Machine Recursive Feature Elimination) is a feature selection technique used in machine learning. It combines the SVM algorithm with a recursive feature elimination process to identify the most relevant features for a given problem.

The basic idea of SVM-RFE is to train an SVM classifier on the entire feature set and then eliminate the least important feature(s) from the set. This process is repeated iteratively until the desired number of features is reached. The importance of each feature is measured by its impact on the performance of the SVM classifier.

SVM-RFE is a powerful method for feature selection because it is able to handle non-linear and high-dimensional data, and it can identify complex relationships between features. It is also able to reduce the overfitting problem by removing redundant or irrelevant features.

We use SVM-RFE pair for feature selection, use the selected features to classify the sample with SVM after the feature selection is completed, and calculate the weight by the obtained slope to obtain the feature importance ranking. In addition, we give some evaluation metrics for machine learning.


## Getting started
Place the file as follows：
PD prediction
|_________SVM-RFE.py
|_________Data.csv

The data in the Data.csv can be referred to the sample in the example. Run the SVM-RFE.py and you will get the result.
We can get the importance ranking of features, the accuracy of classification, recall, precision, ROC graph, and cross-validation scores


## Example output:
Feature sorting results ----------------------
 1) 3-Methoxytyrosine              6.088806
 2) Dopamine 3-o-sulfate           3.941428
 3) DG(18:0_20:4)                  3.615174
 4) GB3(d18:1/24:1)                3.347314
 5) LPE(18:0)                      2.620566
 6) LPC(16:1)                      2.583166
 7) Putrescine                     2.574749
 8) Oxoglutaric acid               2.482240
 9) GlcCer(d18:1/24:1)             2.360064
10) GB3(d18:1/16:0)                2.245248
11) Dopamine 4-o-sulfate           2.203707
12) LPI(18:0)                      2.132594
13) PI(18:0_22:6)                  2.086591
14) TG(22:6_36:2)                  2.062112
15) Sarcosine                      2.006767
16) Piperine                       2.001921
17) Paraxanthine                   1.772677
18) (3-O-sulfo)GalCer(d18:1/24:0(2OH)) 1.763748
19) (3-O-sulfo)GalCer(d18:1/24:1(2OH)) 1.565607
20) PC(40:5)                       1.411093
21) 5-Hydroxytryptophan            0.637601

accuracy:61.48%
precision: 0 : 65.71 %
precision: 1 : 18.52 %
precision: 2 : 78.33 %
recall: 0 : 46.94 %
recall: 1 : 55.56 %
recall: 2 : 73.44 %
![image](https://user-images.githubusercontent.com/102600946/227534718-bd34a1f2-00ca-411f-91a8-58c35f9d70d8.png)
Cross-validation score：
[0.57142857 0.59183673 0.65306122 0.69387755 0.6122449  0.58333333
 0.6875     0.66666667 0.58333333 0.60416667]
0.6247448979591836
