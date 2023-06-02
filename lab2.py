import pandas as pd
from sklearn.naive_bayes import GaussianNB

from datasetHelper import splitDataset, showMetrics, encodeDataset


df = pd.read_csv('datasets/dataset.csv')

encoded_df = encodeDataset(df)

X_train, X_test, y_train, y_test = splitDataset(encoded_df, targetVariable='Disease')

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

showMetrics(encoded_df, y_test, y_pred)
