# FIFABets
1) Download the Kaggle dataset and add to this dir
2) Setup env 
 -Conda env python3 with all the dependencies, which can be seen in import sections of files noted following steps
 -I will add them here later
3) 'python3 Data.py' should clean data
4) 'python train_nn.py' trains

##CS229
Before scripts python3
###Training and Testing
1.Neural Net:
* Train/Test: train.py
* Test on Seperate dataset: predict.py "weights file name" NN
* Class definition: Network.py

2.Logistic Regression:
* Train/Test: lr_pca_test.py
* Test on Seperate dataset: predict.py "weights filename" LR
* Class definition: log_reg_network.py

3.Decision Tree:
* Train/Test: train_DT.py

4.Gradient Boosting:
* Train/Test: train_GB.py

5.Naive Bayes:
* Train/Test: train_NB.py

6.Random Forest:
* Train/Test: train_RFC.py

7.SVM:
* Train/Test: train_SVM.py

8.AdaBoost:
* Train/Test: train_adaBoost.py

9.Q-learning:
* Train: betRL.py "Model type"
* Test: AutoBetter.py "Model type"
* environment definition: env/env.py
