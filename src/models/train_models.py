import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_models(X, y):
    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    lr.fit(X, y)
    rf.fit(X, y)

    return lr, rf


def save_models(lr, rf, output_dir):
    joblib.dump(lr, f"{output_dir}/lr.pkl")
    joblib.dump(rf, f"{output_dir}/rf.pkl")