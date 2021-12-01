import pandas as pd
from sklearn.model_selection import train_test_split

import preprocess_data as pp

raw_df = pp.create_df("../data/EuroMillions_numbers.csv")
new_df = pp.add_data(raw_df, 10)
df = pp.add_binary_winner_column(new_df)
# print(df.iloc[:50])

X = df[["N1","N2","N3","N4","N5","E1","E2"]]
y = df["Winner_binary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # 70% training and 30% test

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=1000)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("F1 score:",metrics.f1_score(y_test, y_pred, average = "binary"))
