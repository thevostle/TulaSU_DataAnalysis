import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV

from datasetHelper import splitDataset, showMetrics, encodeDataset


df = pd.read_csv('datasets/dataset.csv')

encoded_df = encodeDataset(df)

X_train, X_test, y_train, y_test = splitDataset(encoded_df, targetVariable='Disease')

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

y_pred = qda.predict(X_test)

showMetrics(encoded_df, y_test, y_pred)

param_grid = {
    'reg_param': [0.0, 0.1, 0.3, 0.5]
}

grid_search = GridSearchCV(qda, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f'Оптимальные параметры: { grid_search.best_params_ }')
