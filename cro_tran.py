import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import export_text
from sklearn.model_selection import cross_val_score

# Load the crop recommendation data
crop_data = pd.read_csv('crop_recommendation.csv')
print(crop_data['Soil'].value_counts())
print(crop_data['rainfall'].describe())
print(crop_data['label'].value_counts())

# Prepare the features (X) and target (y)
X = crop_data[['Soil', 'rainfall']]
y = crop_data['label']

# Encode categorical variables
X = pd.get_dummies(X)

# Save the column names used for training
columns = X.columns
print("Training columns:", columns)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Evaluate the model
y_pred = dt_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Cross-validation
scores = cross_val_score(dt_model, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy scores: {scores}")
print(f"Mean accuracy: {scores.mean()}")

# Feature importances
importances = dt_model.feature_importances_
for feature, importance in zip(columns, importances):
    print(f"{feature}: {importance}")

# Display decision tree rules
tree_rules = export_text(dt_model, feature_names=list(X.columns))
print(tree_rules)

# Save the trained model and column names to files
with open('recommendation_model.pkl', 'wb') as f:
    pickle.dump(dt_model, f)

with open('columns.pkl', 'wb') as f:
    pickle.dump(columns, f)
