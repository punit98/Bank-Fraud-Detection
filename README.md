# Bank-Fraud-Detection
Detencting Fraud Transactions in a bank. Code and empirical analysis.
Portfolio 5[¶](#Portfolio-5)
============================

1. Import the dataset and explore the data[¶](#1.-Import-the-dataset-and-explore-the-data)
------------------------------------------------------------------------------------------

I am running this script in Google Colab - so you might want to change the location of the dataset and remove the 1st cell.

In [1]:

    from google.colab import drive
    drive.mount('/content/drive')

    ---------------------------------------------------------------------------
    ModuleNotFoundError                       Traceback (most recent call last)
    <ipython-input-1-d5df0069828e> in <module>
    ----> 1 from google.colab import drive
          2 drive.mount('/content/drive')

    ModuleNotFoundError: No module named 'google.colab'

In [2]:

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    data = pd.read_csv("/content/drive/MyDrive/data/bs140513_032310.csv")

    data.head()


Addressing the imbalance in data[¶](#Addressing-the-imbalance-in-data)
----------------------------------------------------------------------

From the data description, we know that there are about 600,000 rows of data and only 7200 fraud transactions which is 1.2% of the data which means that there is a massive imbalance in the data.

Assuming that in a real world situation, the number of fraud transactions is very low, so, at first, I will not oversample the data to remove the imbalance in the data. First I will try classifying algorithms on the original dataset and evaluate the results. After that I will repeat the process after removing the imbalance in the data.

In [3]:

    pd.plotting.scatter_matrix(data, figsize = (10,8))

Out[3]:

    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7f0357936e10>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f0347f19940>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f0347ed0ba8>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f0347e85e10>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f0347ead320>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f0347dee8d0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f0347e21b38>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f0347dd5d68>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f0347dd5dd8>]],
          dtype=object)

![](https://i.ibb.co/vsHKJ4C/Scatter-matrix.png)
In [4]:

    print(data.nunique())

    step             180
    customer        4112
    age                8
    gender             4
    zipcodeOri         1
    merchant          50
    zipMerchant        1
    category          15
    amount         23767
    fraud              2
    dtype: int64

### Inference[¶](#Inference)

According to the kaggle description of the data, the step parameter shows during which step was a particular transaction added to the dataset in the datacollection process. Since it does not add any important information to this dataset, we can safely drop the "step" column.

According to the BankSim paper [BankSim], the "age" column has categortical values with 0 being age \<= 18 and 6 being \>65, and U is unknown value. we will procecss Age column as categorical.

The gender column is also categorical with values "Male", "Female", "Enterprise" and "Unknown"

There are 15 unique values in the "category" column so it will also be treated as categorical data.

The fraud column is the output variable where 0 means that the transaction was not fraud and 1 means that the transaction was fraud.

We see that there only one value in the "zipMerchant" and "zipcodeOri" columns. so we can drop these columns.

There is also extra ' ' as the suffix and prefix in the data, so we need to remove that as well.

2. Datacleaning and preprocessing[¶](#2.-Datacleaning-and-preprocessing)
------------------------------------------------------------------------

First we drop the NaN values in the dataset.

Then I drop the columns which will not be used.

In [5]:

    data = data.dropna()
    data = data.drop(["step", "zipcodeOri", "zipMerchant"],  axis = 1)

    data['customer'] = data['customer'].str.replace("\'", "")
    data['age'] = data['age'].str.replace("\'", "")
    data['gender'] = data['gender'].str.replace("\'", "")
    data['merchant'] = data['merchant'].str.replace("\'", "")
    data['category'] = data['category'].str.replace("\'", "")

    data.head()



Now I will labelencode the columns which have string values, that is, customer, age, gender, merchant and category columns.

And then I will scale the data using StandardScaler.

In [6]:

    from sklearn.preprocessing import LabelEncoder, StandardScaler
    le = LabelEncoder()

    data["customer"] = le.fit_transform(data["customer"])
    data["age"] = le.fit_transform(data["age"])
    data["gender"] = le.fit_transform(data["gender"])
    data["merchant"] = le.fit_transform(data["merchant"])
    data["category"] = le.fit_transform(data["category"])
    data.head()

In [7]:

    data_x = data[data.columns.difference(["fraud"])]
    data_y = data['fraud']

    data_x = pd.get_dummies(data_x, columns=['age', 'gender', 'category'])

    sc = StandardScaler()
    data_x = sc.fit_transform(data_x)
    data_x = pd.DataFrame(data_x)
    data_x.head()



In [8]:

    print(data_x.nunique())

    0     23767
    1      4112
    2        50
    3         2
    4         2
    5         2
    6         2
    7         2
    8         2
    9         2
    10        2
    11        2
    12        2
    13        2
    14        2
    15        2
    16        2
    17        2
    18        2
    19        2
    20        2
    21        2
    22        2
    23        2
    24        2
    25        2
    26        2
    27        2
    28        2
    29        2
    dtype: int64

Now that label-encoding is done, I can one-hot-encode the categorical columns, which are age, gender and category.

Then I will split the data into training set and testing set.


In [10]:

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size = 0.6)

Now that we have a dataset that we can perform operations on, we can proceed with splitting the data into x and y and then into testing and training data.

Now that we have the data split up in different parts, we can apply machiine learning algorithms to the data.

i.) Applying Classification Algorithms[¶](#i.)-Applying-Classification-Algorithms)
----------------------------------------------------------------------------------

### a.) SVC[¶](#a.)-SVC)

In [11]:

    from sklearn.metrics import classification_report

In [12]:

    from sklearn.svm import SVC
    svc = SVC()
    svc.fit(x_train, y_train)
    y_pred_svc = svc.predict(x_test)

    cm_svc = pd.crosstab(y_pred_svc, y_test, rownames = ['pred'], colnames = ['true'], margins = True)
    report_svc = classification_report(y_test, y_pred_svc, labels = [0, 1])
    print(cm_svc, '\n\n', report_svc)

    true       0     1     All
    pred                      
    0     234745   930  235675
    1        245  1938    2183
    All   234990  2868  237858 

                   precision    recall  f1-score   support

               0       1.00      1.00      1.00    234990
               1       0.89      0.68      0.77      2868

        accuracy                           1.00    237858
       macro avg       0.94      0.84      0.88    237858
    weighted avg       0.99      1.00      0.99    237858

SVC takea a lot of time to train the model so we will pass on it in the Cross-Validation data.

### b.) Logistic Regression[¶](#b.)-Logistic-Regression)

In [13]:

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    y_pred_lr = lr.predict(x_test)

    cm_lr = pd.crosstab(y_pred_lr, y_test, rownames = ['pred'], colnames = ['true'], margins = True)
    report_lr = classification_report(y_test, y_pred_lr, labels = [0, 1])
    print(cm_lr, '\n\n', report_lr)

    /usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)

    true       0     1     All
    pred                      
    0     234787  1031  235818
    1        203  1837    2040
    All   234990  2868  237858 

                   precision    recall  f1-score   support

               0       1.00      1.00      1.00    234990
               1       0.90      0.64      0.75      2868

        accuracy                           0.99    237858
       macro avg       0.95      0.82      0.87    237858
    weighted avg       0.99      0.99      0.99    237858

### c.) Random Forest Classifier[¶](#c.)-Random-Forest-Classifier)

In [14]:

    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    y_pred_rfc = rfc.predict(x_test)

    cm_rfc = pd.crosstab(y_pred_rfc, y_test, rownames = ['pred'], colnames = ['true'], margins = True)
    report_rfc = classification_report(y_test, y_pred_rfc, labels = [0, 1])
    print(cm_rfc, '\n\n', report_rfc)

    true       0     1     All
    pred                      
    0     234663   675  235338
    1        327  2193    2520
    All   234990  2868  237858 

                   precision    recall  f1-score   support

               0       1.00      1.00      1.00    234990
               1       0.87      0.76      0.81      2868

        accuracy                           1.00    237858
       macro avg       0.93      0.88      0.91    237858
    weighted avg       1.00      1.00      1.00    237858

### d.) Decision Tree Classifier[¶](#d.)-Decision-Tree-Classifier)

In [15]:

    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    y_pred_dtc = dtc.predict(x_test)

    cm_dtc = pd.crosstab(y_pred_dtc, y_test, rownames = ['pred'], colnames = ['true'], margins = True)
    report_dtc = classification_report(y_test, y_pred_dtc, labels = [0, 1])
    print(cm_dtc, '\n\n', report_dtc)

    true       0     1     All
    pred                      
    0     234196   630  234826
    1        794  2238    3032
    All   234990  2868  237858 

                   precision    recall  f1-score   support

               0       1.00      1.00      1.00    234990
               1       0.74      0.78      0.76      2868

        accuracy                           0.99    237858
       macro avg       0.87      0.89      0.88    237858
    weighted avg       0.99      0.99      0.99    237858

### e.) Ridge Classifier[¶](#e.)-Ridge-Classifier)

In [16]:

    from sklearn.linear_model.ridge import RidgeClassifier
    rc = RidgeClassifier()
    rc.fit(x_train, y_train)
    y_pred_rc = dtc.predict(x_test)

    cm_rc = pd.crosstab(y_pred_rc, y_test, rownames = ['pred'], colnames = ['true'], margins = True)
    report_rc = classification_report(y_test, y_pred_rc, labels = [0, 1])
    print(cm_rc, '\n\n', report_rc)

    /usr/local/lib/python3.6/dist-packages/sklearn/utils/deprecation.py:144: FutureWarning: The sklearn.linear_model.ridge module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.linear_model. Anything that cannot be imported from sklearn.linear_model is now part of the private API.
      warnings.warn(message, FutureWarning)

    true       0     1     All
    pred                      
    0     234196   630  234826
    1        794  2238    3032
    All   234990  2868  237858 

                   precision    recall  f1-score   support

               0       1.00      1.00      1.00    234990
               1       0.74      0.78      0.76      2868

        accuracy                           0.99    237858
       macro avg       0.87      0.89      0.88    237858
    weighted avg       0.99      0.99      0.99    237858

### f.) ADAoost[¶](#f.)-ADAoost)

In [17]:

    from sklearn.ensemble import AdaBoostClassifier
    adab = AdaBoostClassifier()
    adab.fit(x_train, y_train)
    y_pred_adab = adab.predict(x_test)

    cm_adab = pd.crosstab(y_pred_adab, y_test, rownames = ['pred'], colnames = ['true'], margins = True)
    report_adab = classification_report(y_test, y_pred_adab, labels = [0, 1])
    print(cm_adab, '\n\n', report_adab)

    true       0     1     All
    pred                      
    0     234676   852  235528
    1        314  2016    2330
    All   234990  2868  237858 

                   precision    recall  f1-score   support

               0       1.00      1.00      1.00    234990
               1       0.87      0.70      0.78      2868

        accuracy                           1.00    237858
       macro avg       0.93      0.85      0.89    237858
    weighted avg       0.99      1.00      0.99    237858

ii.) Cross-Validation[¶](#ii.)-Cross-Validation)
------------------------------------------------

Now that we have tested some classification models, we can use cross validation on the same algoithms that we used above. I will use 5 folds cross-validation.

In [18]:

    from sklearn.model_selection import KFold, cross_val_predict
    cross_val = KFold(n_splits = 10, random_state= 10, shuffle = False)

    /usr/local/lib/python3.6/dist-packages/sklearn/model_selection/_split.py:296: FutureWarning: Setting a random_state has no effect since shuffle is False. This will raise an error in 0.24. You should leave random_state to its default (None), or set shuffle=True.
      FutureWarning

### a.) Logistic Regression[¶](#a.)-Logistic-Regression)

In [19]:

    lr_cv = LogisticRegression()
    y_pred_lr_cv = cross_val_predict(lr_cv, x_train, y_train, cv = cross_val)
    cm_lr_cv = pd.crosstab(y_pred_lr_cv, y_train, rownames = ['pred'], colnames = ['true'], margins = True)
    report_lr_cv = classification_report(y_train, y_pred_lr_cv, labels = [0, 1])
    print(cm_lr_cv, '\n\n', report_lr_cv)

    /usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    /usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    /usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    /usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    /usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    /usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    /usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    /usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    /usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)

    true       0     1     All
    pred                      
    0     352189  1643  353832
    1        264  2689    2953
    All   352453  4332  356785 

                   precision    recall  f1-score   support

               0       1.00      1.00      1.00    352453
               1       0.91      0.62      0.74      4332

        accuracy                           0.99    356785
       macro avg       0.95      0.81      0.87    356785
    weighted avg       0.99      0.99      0.99    356785

### b.) RFC[¶](#b.)-RFC)

In [20]:

    rfc_cv = RandomForestClassifier()
    y_pred_rfc_cv = cross_val_predict(rfc_cv, x_train, y_train, cv = cross_val)
    cm_rfc_cv = pd.crosstab(y_pred_rfc_cv, y_train, rownames = ['pred'], colnames = ['true'], margins = True)
    report_rfc_cv = classification_report(y_train, y_pred_rfc_cv, labels = [0, 1])
    print(cm_rfc_cv, '\n\n', report_rfc_cv)

    true       0     1     All
    pred                      
    0     351969  1120  353089
    1        484  3212    3696
    All   352453  4332  356785 

                   precision    recall  f1-score   support

               0       1.00      1.00      1.00    352453
               1       0.87      0.74      0.80      4332

        accuracy                           1.00    356785
       macro avg       0.93      0.87      0.90    356785
    weighted avg       1.00      1.00      1.00    356785

### c.) DTC[¶](#c.)-DTC)

In [22]:

    dtc_cv = DecisionTreeClassifier()
    y_pred_dtc_cv = cross_val_predict(dtc_cv, x_train, y_train, cv = cross_val)
    cm_dtc_cv = pd.crosstab(y_pred_dtc_cv, y_train, rownames = ['pred'], colnames = ['true'], margins = True)
    report_dtc_cv = classification_report(y_train, y_pred_dtc_cv, labels = [0, 1])
    print(cm_dtc_cv, '\n\n', report_dtc_cv)

    true       0     1     All
    pred                      
    0     351328  1092  352420
    1       1125  3240    4365
    All   352453  4332  356785 

                   precision    recall  f1-score   support

               0       1.00      1.00      1.00    352453
               1       0.74      0.75      0.75      4332

        accuracy                           0.99    356785
       macro avg       0.87      0.87      0.87    356785
    weighted avg       0.99      0.99      0.99    356785

### d.) Ridge Classifier[¶](#d.)-Ridge-Classifier)

In [23]:

    rc_cv = RidgeClassifier()

### e.) ADABoost[¶](#e.)-ADABoost)

In [24]:

    adab_cv = AdaBoostClassifier()
    y_pred_adab_cv = cross_val_predict(adab_cv, x_train, y_train, cv = cross_val)
    cm_adab_cv = pd.crosstab(y_pred_adab_cv, y_train, rownames = ['pred'], colnames = ['true'], margins = True)
    report_adab_cv = classification_report(y_train, y_pred_adab_cv, labels = [0, 1])
    print(cm_adab_cv, '\n\n', report_adab_cv)

    true       0     1     All
    pred                      
    0     352030  1358  353388
    1        423  2974    3397
    All   352453  4332  356785 

                   precision    recall  f1-score   support

               0       1.00      1.00      1.00    352453
               1       0.88      0.69      0.77      4332

        accuracy                           1.00    356785
       macro avg       0.94      0.84      0.88    356785
    weighted avg       0.99      1.00      0.99    356785

Now since I have also tried the algorithms using cross-validation, I know that there are 30 columns in the x data, I can try using PCA (Principle Data Analysis) to reduce teh number of columns and then I will train all the above algorithms again on the PCA data.

After that I will compare all the metrics using a table and then I will decide on the best model.

In [26]:

    print(data.head())

    cols = ["customer", "age", "gender", "merchant", "category", "amount"]
    data_x_pca = data[data.columns.difference(["fraud"])]
    data_x_pca = data_x_pca[cols]
    print(data_x_pca.head())


    import seaborn as sns
    from sklearn.decomposition import PCA, FactorAnalysis

    pca = PCA()
    data_x_pca = pca.fit(data_x_pca)
    pca_components = pca.components_

    print("explained variance ratio:\n\n", pca.explained_variance_ratio_)

    var_ratio = pd.DataFrame({'var':pca.explained_variance_ratio_,
                 'PC':['pc1','pc2','pc3','pc4','pc5','pc6']})

    sns.barplot(x = 'PC', y = 'var', data=var_ratio, color="c");

       customer  age  gender  merchant  category  amount  fraud
    0       210    4       2        30        12    4.55      0
    1      2753    2       2        30        12   39.68      0
    2      2285    4       1        18        12   26.89      0
    3      1650    3       2        30        12   17.25      0
    4      3585    5       2        30        12   35.72      0
       customer  age  gender  merchant  category  amount
    0       210    4       2        30        12    4.55
    1      2753    2       2        30        12   39.68
    2      2285    4       1        18        12   26.89
    3      1650    3       2        30        12   17.25
    4      3585    5       2        30        12   35.72
    explained variance ratio:

     [9.91188343e-01 8.75286931e-03 5.21648564e-05 5.18593705e-06
     1.25847043e-06 1.78590666e-07]

![](data:image/png;base64,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%0A)

PCA shows that only one variable can capture enough details to contribute to about 90% of the total variance. I will now do factor to reduce the data to 1 column.

Analysis[¶](#Analysis)
======================

# Analysis

Algorithm | Accuracy | True Positive (2840) | True Negative(235018) | 
----------|----------|----------------------|-----------------------|
SVC       |1.00      |1858 (65.42%)         |234882 (99.94)         |
LR        |0.99      |1786 (62.88%)         |234828 (99.91)         |
RFC       |1.00      |2193 (76.46%)         |234729 (99.87)         |
DTC       |0.99      |2141 (75.38%)         |234264 (99.67)         |
Ridge     |0.99      |2141 (75.38%)         |234264 (99.67)         |
ADABoost  |0.99      |1925 (67.78%)         |234728 (99.87)         |


Using Cross-Validation

Algorithm | Accuracy | True Positive (4360) | True Negative(352425) | 
----------|----------|----------------------|-----------------------|
LR        |0.99      |2779 (63.73%)         |352121 (99.91%)        |
RFC       |1.00      |3288 (75.41%)         |351956 (99.86%)        |
SVC       |0.99      |2902 (66.55%)         |352078 (99.90%)        |
DTC       |0.99      |3333 (76.44%)         |351271 (99.67%)        |
Ridge     |0.99      |1544 (35.41%)         |352357 (99.98%)        |
ADABoost  |0.99      |2992 (68.62%)         |351968 (99.87%)        |

Addressing the data Imbalance[¶](#Addressing-the-data-Imbalance)
================================================================

I will address the imbalance in the data by oversampling the fraud values using SMOTE

In [27]:

    data.head()
    data_x_over = data_x
    data_y_over = data_y

    from imblearn.over_sampling import SMOTE

    over = SMOTE()
    data_x_over, data_y_over = over.fit_resample(data_x_over, data_y_over)
    data_y_over = pd.DataFrame(data_y_over)
    print(data_y_over[0].value_counts())

    /usr/local/lib/python3.6/dist-packages/sklearn/externals/six.py:31: FutureWarning: The module is deprecated in version 0.21 and will be removed in version 0.23 since we've dropped support for Python 2.7. Please rely on the official version of six (https://pypi.org/project/six/).
      "(https://pypi.org/project/six/).", FutureWarning)
    /usr/local/lib/python3.6/dist-packages/sklearn/utils/deprecation.py:144: FutureWarning: The sklearn.neighbors.base module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.neighbors. Anything that cannot be imported from sklearn.neighbors is now part of the private API.
      warnings.warn(message, FutureWarning)
    /usr/local/lib/python3.6/dist-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function safe_indexing is deprecated; safe_indexing is deprecated in version 0.22 and will be removed in version 0.24.
      warnings.warn(msg, category=FutureWarning)

    1    587443
    0    587443
    Name: 0, dtype: int64

In [28]:

    x_over_train, x_over_test, y_over_train, y_over_test = train_test_split(data_x_over, data_y_over, train_size =0.7 )

In [34]:

    dtc.fit(x_over_train, y_over_train)
    y_pred_over = dtc.predict(x_over_test)
    y_pred_over = np.array(y_pred_over)
    y_over_test = np.array(y_over_test)
    y_over_test = y_over_test.reshape(-1,)
    print(y_pred_over.shape, y_over_test.shape)

     

    (352466,) (352466,)

In [36]:

    cm_dtc_over = pd.crosstab(y_pred_over, y_over_test, rownames = ['pred'], colnames = ['true'], margins = True)
    report_dtc_over = classification_report(y_over_test, y_pred_over, labels = [0, 1])
    print( cm_dtc_over, '\n\n', report_dtc_over)

    true       0       1     All
    pred                        
    0     174328    1676  176004
    1       2105  174357  176462
    All   176433  176033  352466 

                   precision    recall  f1-score   support

               0       0.99      0.99      0.99    176433
               1       0.99      0.99      0.99    176033

        accuracy                           0.99    352466
       macro avg       0.99      0.99      0.99    352466
    weighted avg       0.99      0.99      0.99    352466

Here we can see that after oversampling the data, RFC gives correct prediction of 99% of both the fraud and not fraud transactions.

Conclusion[¶](#Conclusion)
==========================

If we use oversampling to remove the data, then we can easily predict the fraud and non-fraud transactions with an accuracy of 99% using RFC.

However, if we do not use oversampling, then RFC gives the best results with prediction oaccuracy of 99% for non-fraud transactions and 76.44% for fraud transactions.

In [ ]:

     
