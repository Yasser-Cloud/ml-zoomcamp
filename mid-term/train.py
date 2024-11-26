import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd 

# Read the data
data  = pd.read_csv('ks-projects-201801.csv', encoding='latin1')

data['deadline'] = pd.to_datetime(data['deadline'])
data['launched'] = pd.to_datetime(data['launched'])


data['campaign_duration'] = (data['deadline'] - data['launched']).dt.days

data['launched_dayofweek'] = data['launched'].dt.dayofweek
data['launched_month'] = data['launched'].dt.month

numerical_features = [ 'backers', 'usd_pledged_real','usd_goal_real','launched_month','campaign_duration']
categorical_features = ['main_category', 'currency', 'country']
# Train-test split
df_train_full, df_test = train_test_split(data, test_size=0.2, stratify=data['state'], random_state=42,)

# Reset indices
df_train_full = df_train_full.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Separate target variable
y_train = df_train_full.state.values
y_test = df_test.state.values

# Remove target column from features
del df_train_full['state']
del df_test['state']

train_dict = df_train_full[categorical_features + numerical_features].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
dv.fit(train_dict)

X_train = dv.transform(train_dict)
# test
test_dict = df_test[categorical_features + numerical_features].to_dict(orient='records')

X_test = dv.transform(test_dict)
# Train the Random Forest model
model = RandomForestClassifier(max_depth=20, n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

print(classification_report(y_test, y_pred))

# Save the model and vectorizer using Pickle
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(dv, vectorizer_file)

print("Model and vectorizer saved.")



