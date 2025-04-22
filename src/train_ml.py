import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# مثال لسكري
df = pd.read_csv('data/pima-indians-diabetes.csv')
X = df.drop('Outcome', axis=1); y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'RF': RandomForestClassifier(n_estimators=100),
    'DT': DecisionTreeClassifier(),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'LR': LogisticRegression()
}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{name} CV Accuracy: {scores.mean():.3f}")
    model.fit(X_train, y_train)
    joblib.dump(model, f'models/{name.lower()}_diabetes.pkl')

