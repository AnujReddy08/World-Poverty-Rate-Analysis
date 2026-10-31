# 1. Import Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix


# 2. Load Dataset

df = pd.read_csv("WB_PIP.csv")

print(df.head())
print(df.info())
print(df.describe())


# 3. Data Cleaning

df = df[df['INDICATOR_LABEL'].str.contains(
    'poverty headcount', case=False, na=False)]

df['OBS_VALUE'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')

df = df.dropna(subset=['OBS_VALUE'])

df = df[df['OBS_VALUE'].between(0, 100)]

df = df.drop_duplicates()

print("\nCleaned Data")
print(df.shape)

print("\nMissing Values")
print(df.isnull().sum())


# 4. Select Features

X = df[['REF_AREA_LABEL', 'TIME_PERIOD']]

y = df['OBS_VALUE']

X = pd.get_dummies(X, drop_first=True)


# 5. Data Visualization

sns.histplot(df['OBS_VALUE'], kde=True)
plt.title("Poverty Rate Distribution")
plt.xlabel("Poverty Rate")
plt.show()


country = df.groupby('REF_AREA_LABEL')['OBS_VALUE'].mean()
country = country.sort_values(ascending=False).head(10)

country.plot(kind='bar')
plt.title("Top 10 Countries by Poverty Rate")
plt.ylabel("Poverty Rate")
plt.show()


year = df.groupby('TIME_PERIOD')['OBS_VALUE'].mean()

year.plot(kind='line', marker='o')
plt.title("Poverty Rate Over Years")
plt.xlabel("Year")
plt.ylabel("Poverty Rate")
plt.show()


# 6. Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 7. Normalization

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# =====================================
# REGRESSION
# =====================================

print("\nREGRESSION MODELS")


# 8. Linear Regression

model1 = LinearRegression()

model1.fit(X_train, y_train)

pred1 = model1.predict(X_test)


# 9. Decision Tree Regression

model2 = DecisionTreeRegressor(random_state=42)

model2.fit(X_train, y_train)

pred2 = model2.predict(X_test)


# 10. Random Forest Regression

model3 = RandomForestRegressor(n_estimators=100, random_state=42)

model3.fit(X_train, y_train)

pred3 = model3.predict(X_test)


# 11. Regression Evaluation

models = [pred1, pred2, pred3]

names = ['Linear Regression', 'Decision Tree', 'Random Forest']

for i in range(3):

    print("\n", names[i])

    print("MAE:", mean_absolute_error(y_test, models[i]))

    print("MSE:", mean_squared_error(y_test, models[i]))

    print("RMSE:", np.sqrt(mean_squared_error(y_test, models[i])))

    print("R2 Score:", r2_score(y_test, models[i]))


# 12. Regression Graph

plt.scatter(y_test, pred3)

plt.xlabel("Actual Poverty Rate")
plt.ylabel("Predicted Poverty Rate")

plt.title("Random Forest Regression")

plt.show()


# =====================================
# CLASSIFICATION
# =====================================

print("\nCLASSIFICATION MODELS")


# 13. Create Categories

df['Poverty_Category'] = pd.cut(
    df['OBS_VALUE'],
    bins=[-0.01, 20, 40, 100],
    labels=['Low', 'Moderate', 'High']
)

print("\nCategory Values")
print(df['Poverty_Category'].value_counts())


# 14. Select Features and Target

X = df[['REF_AREA_LABEL', 'TIME_PERIOD']]

y = df['Poverty_Category']

X = pd.get_dummies(X, drop_first=True)


# 15. Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 16. Normalization

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# 17. Logistic Regression

model4 = LogisticRegression(max_iter=1000, class_weight='balanced')

model4.fit(X_train, y_train)

pred4 = model4.predict(X_test)


# 18. KNN Classification

model5 = KNeighborsClassifier(n_neighbors=5)

model5.fit(X_train, y_train)

pred5 = model5.predict(X_test)


# 19. Random Forest Classification

model6 = RandomForestClassifier(n_estimators=100, random_state=42,
                                class_weight='balanced')

model6.fit(X_train, y_train)

pred6 = model6.predict(X_test)


# 20. Classification Evaluation

models = [pred4, pred5, pred6]

names = ['Logistic Regression', 'KNN', 'Random Forest']

for i in range(3):

    print("\n", names[i])

    print("Accuracy:", accuracy_score(y_test, models[i]))

    print("Precision:", precision_score(
        y_test, models[i], average='weighted', zero_division=0))

    print("Recall:", recall_score(
        y_test, models[i], average='weighted', zero_division=0))

    print("F1 Score:", f1_score(
        y_test, models[i], average='weighted', zero_division=0))


# 21. Confusion Matrix

for i in range(3):

    cm = confusion_matrix(
        y_test, models[i],
        labels=['Low', 'Moderate', 'High']
    )

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    plt.title(names[i] + " Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.show()
