import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Folder paths
DATA_DIR = "Data"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# NSL-KDD column names
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty"
]


print("Loading dataset...")

train_path = os.path.join(DATA_DIR, "KDDTrain+.txt")
test_path = os.path.join(DATA_DIR, "KDDTest+.txt")

train_df = pd.read_csv(train_path, names=columns)
test_df = pd.read_csv(test_path, names=columns)

print("Training data shape:", train_df.shape)
print("Testing data shape:", test_df.shape)


# Convert labels into Normal and Attack
train_df["label"] = train_df["label"].apply(lambda x: "Normal" if x == "normal" else "Attack")
test_df["label"] = test_df["label"].apply(lambda x: "Normal" if x == "normal" else "Attack")


# Drop difficulty column
train_df = train_df.drop("difficulty", axis=1)
test_df = test_df.drop("difficulty", axis=1)


# Class distribution graph
plt.figure(figsize=(6, 4))
sns.countplot(x=train_df["label"])
plt.title("NSL-KDD Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"))
plt.close()


# Features and target
X_train = train_df.drop("label", axis=1)
y_train = train_df["label"]

X_test = test_df.drop("label", axis=1)
y_test = test_df["label"]


# Combine train and test before encoding so columns match
combined = pd.concat([X_train, X_test], axis=0)

combined_encoded = pd.get_dummies(combined)

X_train_encoded = combined_encoded.iloc[:len(X_train), :]
X_test_encoded = combined_encoded.iloc[len(X_train):, :]


# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)


# Decision Tree model
print("\nTraining Decision Tree...")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)

dt_predictions = dt_model.predict(X_test_scaled)
dt_accuracy = accuracy_score(y_test, dt_predictions)

print("Decision Tree Accuracy:", dt_accuracy)
print(classification_report(y_test, dt_predictions))


# Random Forest model
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

rf_predictions = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print("Random Forest Accuracy:", rf_accuracy)
print(classification_report(y_test, rf_predictions))


# Confusion matrix for Random Forest
cm = confusion_matrix(y_test, rf_predictions, labels=["Normal", "Attack"])

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Attack"],
    yticklabels=["Normal", "Attack"]
)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()


# Model comparison graph
model_names = ["Decision Tree", "Random Forest"]
accuracies = [dt_accuracy, rf_accuracy]

plt.figure(figsize=(6, 4))
bars = plt.bar(model_names, accuracies)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom"
    )

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"))
plt.close()


# Feature importance for Random Forest
feature_importance = pd.Series(rf_model.feature_importances_, index=X_train_encoded.columns)

plt.figure(figsize=(8, 6))
feature_importance.nlargest(15).sort_values().plot(kind="barh")
plt.title("Top 15 Important Features")
plt.xlabel("Importance")
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
plt.close()


# Save best model and required files
joblib.dump(rf_model, os.path.join(MODEL_DIR, "ids_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(X_train_encoded.columns.tolist(), os.path.join(MODEL_DIR, "columns.pkl"))

print("\nTraining completed successfully!")
print("Model saved inside models folder.")
print("Graphs saved inside outputs folder.")