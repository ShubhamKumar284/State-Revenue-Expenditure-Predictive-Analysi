import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"C:\Users\shubh\Downloads\7567_source_data.csv")
print(data.info())
print(data.shape)

df = data.copy()
df = df.drop_duplicates()
df = df.fillna(method='ffill')
df = df[df["Budget type"] == "Accounts"]
df["srcYear"] = df["srcYear"].astype(str)

# Select required columns
df = df[
    [
        "srcYear",
        "srcStateName",
        "Total expenditure",
        "Education, sports, art and culture",
        "Medical and public health",
        "Agriculture and allied activities",
        "Energy",
        "Transport and communications"
    ]
]

df = df.dropna()

print("\nAfter Transformation")
print(df.info())
clean_path = r"C:\Users\shubh\Downloads\Revenue_Expenditure_Cleaned.csv"
df.to_csv(clean_path, index=False)


# EXPLORATORY DATA ANALYSIS (EDA)
print(df.columns)                  # Column names
print(df.shape)                    # Number of rows and columns
print(df.info())                   # Data types and non-null counts
print(df.describe())                 # Summary for numerical columns
print(df.head())                   # First five records
print(df.tail())                   # Last five records
print(df.isnull().sum())           # Total missing values per column
print(df.duplicated().sum())       # Check for duplicate rows
print(df.dropna())                 # Remove missing/duplicate values
print(df.fillna(value=0))   # Fill missing/duplicate values



# # OBJECTIVE 1 – To predict state-wise revenue expenditure using regression techniques based on 
# # historical budget data.
# # Simple & Multiple Linear Regression

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix

X = df[["Education, sports, art and culture", "Medical and public health"]]
y = df["Total expenditure"]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X = pd.DataFrame(scaler_X.fit_transform(X), columns=X.columns)
y = scaler_y.fit_transform(y.values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))


threshold = y_test.mean()

y_test_class = (y_test >= threshold).astype(int)
y_pred_class = (y_pred >= threshold).astype(int)

acc = accuracy_score(y_test_class, y_pred_class)
print("Accuracy:", acc*100, "%")

cm = confusion_matrix(y_test_class, y_pred_class)
print("Confusion Matrix:\n", cm)

# Confusion Matrix Heatmap
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix – Revenue Expenditure")
plt.show()


# Actual vs Predicted Scatterplot
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, color='purple')
plt.plot([0,1],[0,1],'r--')
plt.xlabel("Actual Revenue Expenditure")
plt.ylabel("Predicted Revenue Expenditure")
plt.title("Actual vs Predicted Revenue Expenditure")
plt.grid(True)
plt.show()



residuals = y_test - y_pred

# Residual Plot
plt.figure(figsize=(8,5))
plt.scatter(y_pred, residuals, color='blue')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot – Multiple Linear Regression")
plt.grid(True)
plt.show()


# Feature Relationship Scatterplot
plt.figure(figsize=(8,5))
sns.scatterplot(x=X.iloc[:,0], y=y.flatten(), label="Education")
sns.scatterplot(x=X.iloc[:,1], y=y.flatten(), label="Medical")
plt.xlabel("Normalized Feature Values")
plt.ylabel("Normalized Total Expenditure")
plt.title("Feature Relationship with Total Expenditure")
plt.legend()
plt.show()



# # OBJECTIVE 2 – To classify states as over-spending or under-spending 
# # based on budget estimates and actual revenue expenditure.
# # Logistic Regression & KNN


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud



df["Spend_Class"] = np.where(
    df["Total expenditure"] > df["Total expenditure"].mean(), 1, 0
)

X = df[["Education, sports, art and culture", "Medical and public health"]]
y = df["Spend_Class"]

scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

print("\nLogistic Regression Accuracy:", accuracy_score(y_test, log_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, log_pred))
print("\nClassification Report:\n", classification_report(y_test, log_pred))
print("\nLogistic Regression Precision:", precision_score(y_test, log_pred))
print("\nLogistic Regression Recall:", recall_score(y_test, log_pred))
print("\nLogistic Regression F1-Score:", f1_score(y_test, log_pred))


# Confusion Matrix Heatmap
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, log_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix – Logistic Regression")
plt.show()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

print("\nKNN Accuracy:", accuracy_score(y_test, knn_pred)*100, "%")
print("\nKNN Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))
print("\nKNN Precision:", precision_score(y_test, knn_pred))
print("\nKNN Recall:", recall_score(y_test, knn_pred))
print("\nKNN F1-Score:", f1_score(y_test, knn_pred))


# KNN Confusion Matrix Scatterplot
plt.figure(figsize=(8,5))
sns.scatterplot(
    x=X.iloc[:,0],
    y=X.iloc[:,1],
    hue=y,
    palette="Set1",
    alpha=0.6
)
plt.xlabel("Education Expenditure (Normalized)")
plt.ylabel("Medical Expenditure (Normalized)")
plt.title("Feature Distribution by Spend Class")
plt.legend(title="Spend Class (0=Low, 1=High)")
plt.show()


# Spend Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Distribution of Spend Classes")
plt.xlabel("Spend Class")
plt.ylabel("Count")
plt.show()

# Word Cloud for High Expenditure States
high_spend_states = df[df["Spend_Class"] == 1]["srcStateName"]
text = " ".join(high_spend_states.astype(str))

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='viridis'
).generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud – High Expenditure States")
plt.show()





# # OBJECTIVE 3 – STo compare the performance of tree-based and 
# # margin-based classifiers in predicting expenditure behavior of states.
# # Decision Tree, Support Vector Machine (SVM)

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Prediction
dt_pred = dt_model.predict(X_test)

print("\nDecision Tree Accuracy:", accuracy_score(y_test, dt_pred)*100, "%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, dt_pred))
print("\nClassification Report:\n", classification_report(y_test, dt_pred))
print("\nDecision Tree Precision:", precision_score(y_test, dt_pred))
print("\nDecision Tree Recall:", recall_score(y_test, dt_pred))
print("\nDecision Tree F1-Score:", f1_score(y_test, dt_pred))


# Confusion Matrix Heatmap
plt.figure(figsize=(5,4))
sns.heatmap(
    confusion_matrix(y_test, dt_pred),
    annot=True,
    fmt='d',
    cmap='Greens'
)

# Decision Tree Confusion Matrix
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix – Decision Tree")
plt.show()

plt.figure(figsize=(14,6))
plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=["Low Spend", "High Spend"],
    filled=True
)
plt.title("Decision Tree – Revenue Expenditure Classification")
plt.show()

feature_importance = pd.Series(
    dt_model.feature_importances_,
    index=X.columns
)


# Feature Importance Bar Plot
plt.figure(figsize=(6,4))
feature_importance.sort_values().plot(kind='barh', color='orange')
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance – Decision Tree")
plt.grid(True)
plt.show()




# # OBJECTIVE 4 – To identify patterns among states by clustering them based 
# # on revenue expenditure characteristics.
# # K-Means Clustering, Hierarchical Clustering

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

cluster_data = df[[
    "Education, sports, art and culture",
    "Medical and public health",
    "Agriculture and allied activities"
]]

scaler = MinMaxScaler()
cluster_data = pd.DataFrame(
    scaler.fit_transform(cluster_data),
    columns=cluster_data.columns
)

wcss = []     # within-cluster sum of squares

for i in range(1, 8):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(cluster_data)
    wcss.append(kmeans.inertia_)       # inertaia_ gives WCSS value


# Elbow Method Plot
plt.figure(figsize=(6,4))
plt.plot(range(1,8), wcss, marker='o')
plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
cluster_data["Cluster"] = kmeans.fit_predict(cluster_data)


# K-Means Clustering Scatterplot
plt.figure(figsize=(7,5))
sns.scatterplot(
    x="Education, sports, art and culture",
    y="Medical and public health",
    hue="Cluster",
    data=cluster_data,
    palette="Set1",
    alpha=0.7
)
plt.title("K-Means Clustering of States (Education vs Medical)")
plt.xlabel("Education Expenditure (Normalized)")
plt.ylabel("Medical Expenditure (Normalized)")
plt.legend(title="Cluster")
plt.show()


# Pairplot of Clusters
sns.pairplot(cluster_data, hue="Cluster", palette="Set1")
plt.suptitle("Pairwise Feature Relationships by Cluster", y=1.02)
plt.show()


# Cluster-wise Average Expenditure Bar Plot
cluster_means = cluster_data.groupby("Cluster").mean()
cluster_means.plot(kind='bar', figsize=(8,5))
plt.title("Cluster-wise Average Sector Expenditure")
plt.xlabel("Cluster")
plt.ylabel("Average Normalized Expenditure")
plt.grid(True)
plt.show()
