import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from datasetHelper import splitDataset, showMetrics, encodeDataset


df = pd.read_csv('datasets/dataset.csv')

encoded_df = encodeDataset(df)

X_train, X_test, y_train, y_test = splitDataset(encoded_df, targetVariable='Disease')

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

showMetrics(encoded_df, y_test, y_pred)

param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 1, 3],
    'min_samples_split': [1, 3, 5],
    'min_samples_leaf': [1, 3, 5]
}

grid_search = GridSearchCV(rf, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f'Оптимальные параметры: { grid_search.best_params_ }')
