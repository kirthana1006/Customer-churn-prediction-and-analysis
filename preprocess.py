import pandas as pd
df = pd.read_csv("Telco_Customer_Churn_Dataset .csv")
from sklearn.preprocessing import LabelEncoder
print(df.isnull().sum())
df = df.drop_duplicates()
print(df)
print(df.info())
print(df.duplicated().sum())
print(df.dtypes)
print(df.select_dtypes(include=['object']).columns)
print(df["TotalCharges"].dtype)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(df["TotalCharges"].dtype)
print(df["TotalCharges"].isnull().sum())
df = df.drop("customerID", axis=1)
print(df.columns)
encoder = LabelEncoder()
binary_columns = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn"
]
for column in binary_columns:
    df[column] = encoder.fit_transform(df[column])
print(df.head())
multi_columns = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod"
]
df = pd.get_dummies(df, columns=multi_columns)
print(df.head())
print(df.info())
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
print(df["TotalCharges"].isnull().sum())
print(df.columns)
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
print(df.info())
print(df.isnull().sum())
df.to_csv("cleaned_churn.csv", index=False)
print("Task 1:Data preprocessing completed successfully!")
X = df.drop("Churn", axis=1)
y = df["Churn"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Actual Values:")
print(y_test.head())
print("\nPredicted Values:")
print(y_pred[:5])
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, y_pred)
print("ROC-AUC Score:", round(auc, 2))
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title("Confusion Matrix")
plt.show()
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("ROC Curve")
plt.show()
print("Task 2:Training and testing completed successfully!")
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})
feature_importance["Absolute Coefficient"] = feature_importance["Coefficient"].abs()
feature_importance = feature_importance.sort_values(
    by="Absolute Coefficient",
    ascending=False
)
print(feature_importance)
print(feature_importance.head(10))
top10 = feature_importance.head(10)
plt.figure(figsize=(10,6))
plt.barh(top10["Feature"], top10["Absolute Coefficient"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Features Influencing Customer Churn")
plt.gca().invert_yaxis()
plt.show()
print("Task 3: Feature selection and relevance analysis completed successfully!")
lr_model = LogisticRegression(max_iter=5000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print("Logistic Regression Accuracy:", round(lr_accuracy * 100, 2), "%")
print("Since, the logistic regression technique has high accuracy (82.11%), we use it for binary classification.")
print("Task 4: Model selection completed successfully!")
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
print("Task 5: Model has been trained successfully!")
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,classification_report)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
print("_______Model Evaluation Results_______")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print("Task 6: Model has been evaluated and project is completed successfully!")