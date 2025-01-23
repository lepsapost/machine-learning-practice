from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#   load the iris dataset
data = load_iris()
X = data.data
y = data.target

#   split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

#   create a decision tree classifier model
tree_model = DecisionTreeClassifier(random_state=42)

#   train the model
tree_model.fit(X_train, y_train)

#   make predictions on the test data
y_pred_tree = tree_model.predict(X_test)

#   evaluate accuracy
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f"Decision Tree Model Accuracy: {accuracy_tree}")