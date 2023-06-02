import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("datasets/dataset.csv")
df_severity = pd.read_csv("datasets/severity.csv")

"""
1.	Изучить распределение целевых классов и нескольких категориальных признаков.
"""
def task_1():
    class_counts = df['Disease'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title('Распределение целевого класса')
    plt.xlabel('Disease')
    plt.ylabel('Количество')
    plt.xticks(rotation=90)
    plt.show()

    symptom1_counts = df['Symptom_1'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=symptom1_counts.index, y=symptom1_counts.values)
    plt.title('Распределение Symptom_1')
    plt.xlabel('Symptom_1')
    plt.ylabel('Количество')
    plt.xticks(rotation=90)
    plt.show()

"""
2.	Нарисовать распределения нескольких числовых признаков.
"""
def task_2():
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

    axes[0].hist(df_severity['weight'], bins=20, color='skyblue')
    axes[0].set_title('Распределение weight')
    axes[0].set_xlabel('Значение')
    axes[0].set_ylabel('Частота')

    plt.tight_layout()
    plt.show()


"""
3.	Произвести нормализацию нескольких числовых признаков.
"""
def task_3():
    numeric_features = ['weight']

    scaler = StandardScaler()

    df_severity[numeric_features] = scaler.fit_transform(df_severity[numeric_features])
    df_severity.to_csv('datasets/severity_normalize.csv', index=False)

"""
4.	Посмотреть и визуализировать корреляцию признаков.
"""
def task_4():
    encoded_df = df.copy()

    encoded_df_disease = pd.get_dummies(encoded_df, columns=["Disease"])
    print(encoded_df_disease.head())
    print(list(encoded_df_disease.columns.values)[17:])

    encoded_df_symptom = pd.get_dummies(encoded_df, columns=["Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4", "Symptom_5", "Symptom_6","Symptom_7", "Symptom_8","Symptom_9", "Symptom_10","Symptom_11", "Symptom_12","Symptom_13", "Symptom_14", "Symptom_15", "Symptom_16", "Symptom_17" ])
    print(encoded_df_symptom.head())

    correlationsList = []

    class correlationItem:
        def __init__(self, target_1, target_2, value):
            self.target_1 = target_1
            self.target_2 = target_2
            self.value = value

    for disease in list(encoded_df_disease.columns.values)[17:]:
        for symptom in list(encoded_df_symptom.columns.values)[1:]:
            correlation = encoded_df_disease[disease].corr(encoded_df_symptom[symptom])
            correlationsList.append(correlationItem(disease, symptom, correlation))

    for i in sorted(correlationsList, key=lambda x: x.value):
        print(f"{i.target_1} / {i.target_2} : {i.value}")


if (__name__ == '__main__'):
    task_1()
    task_2()
    task_3()
    task_4()
