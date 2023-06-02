import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score


def encodeDataset(dataset):
    try:
        dataset
    except NameError:
        return 0
    
    encoded_df = dataset.copy()

    for column in encoded_df.columns:
        le = LabelEncoder()
        encoded_df[column] = le.fit_transform(encoded_df[column])

    return encoded_df

def splitDataset(dataset, targetVariable, printDatasets=False):
    try:
        dataset, targetVariable
    except NameError:
        return 0

    df = dataset
    X = df.drop(targetVariable, axis=1)
    y = df[targetVariable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if (printDatasets):
        print("Размер выборки (X):", X.shape)
        print("Размер выборки (y):", y.shape)

        print(f"Размер обучающей выборки (X_train): { X_train.shape }")
        print(f"Размер тестовой выборки (X_test): { X_test.shape }")
        print(f"Размер обучающей выборки (y_train): { y_train.shape }")
        print(f"Размер тестовой выборки (y_test): { y_test.shape }", '\n')

    return X_train, X_test, y_train, y_test

def showMetrics(dataset, y_test, y_pred):
    try:
        dataset, y_test, y_pred
    except NameError:
        return 0

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    correlation_matrix = dataset.corr()

    print(f"Accuracy: { accuracy }")
    print(f"Precision: { precision }")
    print('\n', f'Корреляционная матрица:\n{ correlation_matrix }')

    return accuracy, precision, correlation_matrix
