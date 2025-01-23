from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#   load the iris dataset
data = load_iris()
X = data.data
y = data.target


#   split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)


#   creating a logistic regression model
model = LogisticRegression(max_iter=200)


#   train the model
model.fit(X_train, y_train)


#   make predictions on the test data
y_pred = model.predict(X_test)


#   evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")