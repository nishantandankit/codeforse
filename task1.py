
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Number of samples in training set: {len(X_train)}")
print(f"Number of samples in testing set: {len(X_test)}")

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
print("\nDecision Tree model trained successfully!")

y_pred_test = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_test)
print(f"\nAccuracy of the model on the test set: {accuracy:.2f}")

plt.figure(figsize=(15, 10))
plot_tree(model,
          filled=True,
          rounded=True,
          class_names=target_names,
          feature_names=feature_names,
          fontsize=10)
plt.title("Decision Tree for Iris Classification (max_depth=3)")
plt.show()

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

sorted_feature_names = [feature_names[i] for i in indices]

print("\nFeature Importances:")
for i in indices:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

plt.figure(figsize=(10, 6))
plt.title("Feature Importances from Decision Tree")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), sorted_feature_names, rotation=45, ha="right")
plt.ylabel("Importance Score")
plt.xlabel("Feature")
plt.tight_layout()
plt.show()


print("\n--- Basic Analysis ---")
print("The plot above shows the decision tree.")
print("- Each box (node) shows a condition (e.g., 'petal width (cm) <= 0.8').")
print("- If the condition is true, you go left; if false, you go right.")
print("- 'samples' tells you how many data points reached that node.")
print("- 'value' shows how many samples belong to each class [setosa, versicolor, virginica] at that node.")
print("- 'class' is the predicted class for samples reaching that node.")
print("- Leaf nodes (nodes at the bottom with no further splits) are the final predictions.")
print(f"The model predicted with an accuracy of {accuracy*100:.2f}% on the unseen test data.")
print("\nThe 'Feature Importances' bar graph shows which measurements the tree used most to make decisions.")
print("A higher bar means that feature was more important for the classification.")
