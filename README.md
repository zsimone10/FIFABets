# FIFABets

##CS229

###Collecting and Cleaning Data
1.Download the sqlite database from https://www.kaggle.com/kvnchn/notebook399529d6ab/data

2.Run the cells in Data.ipynb
* Extracts associates match and team attribute data from the above database
* Drops sparse rows and unnecessary columns
* Writes processed data to CSV files

3.Run the cells in scrape_team_attributes.ipynb
* First, you may need to download historical betting odds from
  football-data.co.uk & name the CSVs appropriately
* Scrapes team attributes from SOFIFA.com for recent EPL seasons
* Associates them with historical betting odds and match results from
  football-data.co.uk

4.Run the cells in fill_normalize_PCA.ipynb
* Fills missing betting odds using average of present betting odds
* Fills missing team attribute data using K-nearest neighbors
* Normalizes data
* Reduces dimensionality with PCA

5.visualize_data.ipynb
* For creating charts, tables, and visuals

####Training and Testing
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
