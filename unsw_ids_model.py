import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle

df = pd.read_csv("unsw_nb15_labeled.csv")

drop_cols = ['flow_id', 'source_ip', 'destination_ip']
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

print("\nFull Dataset Category Counts:")
print(df["attack_label"].value_counts())
print("\nBinary Label Counts:")
print(df["binary_label"].value_counts())

print("\n--- BINARY CLASSIFICATION ---")
df_bin = df.copy()
df_bin['binary_label'] = df_bin['binary_label'].astype(int)

min_class_size = df_bin['binary_label'].value_counts().min()
df_bin_balanced = pd.concat([
    df_bin[df_bin['binary_label'] == 0].sample(min_class_size, random_state=42),
    df_bin[df_bin['binary_label'] == 1].sample(min_class_size, random_state=42)
])

for col in df_bin_balanced.select_dtypes(include='object').columns:
    df_bin_balanced[col] = LabelEncoder().fit_transform(df_bin_balanced[col].astype(str))

X_bin = df_bin_balanced.drop(['attack_label', 'binary_label'], axis=1)
y_bin = df_bin_balanced['binary_label']
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X_bin, y_bin, test_size=0.3, stratify=y_bin, random_state=42)

clf_bin = RandomForestClassifier(n_estimators=100, random_state=42)
clf_bin.fit(X_train_bin, y_train_bin)
y_pred_bin = clf_bin.predict(X_test_bin)

print("\nBinary Classification Report:")
print(classification_report(y_test_bin, y_pred_bin))

cm_bin = confusion_matrix(y_test_bin, y_pred_bin)
sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Blues')
plt.title("Binary Classification Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\n--- MULTICLASS CLASSIFICATION ---")

N = 2000
df_multi = df.groupby("attack_label", group_keys=False).apply(lambda x: x.sample(min(N, len(x)), random_state=42))

print("\nSampled Attack Label Counts:")
print(df_multi["attack_label"].value_counts())

X_multi = df_multi.drop(['attack_label', 'binary_label'], axis=1)
y_multi = df_multi['attack_label']

for col in X_multi.select_dtypes(include='object').columns:
    X_multi[col] = LabelEncoder().fit_transform(X_multi[col].astype(str))

le_multi = LabelEncoder()
y_multi_encoded = le_multi.fit_transform(y_multi)

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi_encoded, stratify=y_multi_encoded, test_size=0.3, random_state=42
)

clf_multi = RandomForestClassifier(n_estimators=100, random_state=42)
clf_multi.fit(X_train_multi, y_train_multi)
y_pred_multi = clf_multi.predict(X_test_multi)

print("\nMulticlass Classification Report:")
print(classification_report(y_test_multi, y_pred_multi, target_names=le_multi.classes_))

cm_multi = confusion_matrix(y_test_multi, y_pred_multi)
plt.figure(figsize=(14, 10))
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='YlGnBu', xticklabels=le_multi.classes_, yticklabels=le_multi.classes_)
plt.title("Multiclass Classification Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

binary_acc = clf_bin.score(X_test_bin, y_test_bin)
multi_acc = clf_multi.score(X_test_multi, y_test_multi)

plt.figure(figsize=(6, 4))
plt.bar(["Binary", "Multiclass"], [binary_acc, multi_acc], color=["blue", "green"])
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
