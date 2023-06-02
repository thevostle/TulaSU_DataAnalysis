import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from datasetHelper import splitDataset, showMetrics, encodeDataset


df = pd.read_csv('datasets/dataset.csv')

encoded_df = encodeDataset(df)

X_train, X_test, y_train, y_test = splitDataset(encoded_df, targetVariable='Disease')

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

showMetrics(encoded_df, y_test, y_pred)

param_grid = {
    'C': [0.1, 0.5, 1, 5, 10],
    'penalty': ['l1', 'l2']
}

grid_search = GridSearchCV(logreg, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f'Оптимальные параметры: { grid_search.best_params_ }')
