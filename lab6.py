import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from datasetHelper import splitDataset, showMetrics, encodeDataset


df = pd.read_csv('datasets/dataset.csv')

encoded_df = encodeDataset(df)

X_train, X_test, y_train, y_test = splitDataset(encoded_df, targetVariable='Disease')

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

showMetrics(encoded_df, y_test, y_pred)

param_grid = {
    'max_depth': [None, 1, 3, 5],
    'min_samples_split': [1, 3, 5],
    'min_samples_leaf': [1, 3, 5]
}

grid_search = GridSearchCV(dt, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f'Оптимальные параметры: { grid_search.best_params_ }')
