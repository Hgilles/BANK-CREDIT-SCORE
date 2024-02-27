import joblib
def predict(data):
    lr=joblib.load("rf_model.sav")
    return lr.predict(data)