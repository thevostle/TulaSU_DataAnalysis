import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from datasetHelper import splitDataset, showMetrics, encodeDataset


df = pd.read_csv('datasets/dataset.csv')

encoded_df = encodeDataset(df)

X_train, X_test, y_train, y_test = splitDataset(encoded_df, targetVariable='Disease')

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

showMetrics(encoded_df, y_test, y_pred)

param_grid = {
    'n_neighbors': [3, 5, 10],
    'weights': ['uniform', 'distance']
}

grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f'Оптимальные параметры: { grid_search.best_params_ }')
