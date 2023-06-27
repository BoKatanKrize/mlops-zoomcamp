import os

import mlflow
from flask import Flask, request, jsonify


mlflow.set_tracking_uri("http://127.0.0.1:5000")

RUN_ID = '068bda10a3ed4b73a771df771161f60a'
# RUN_ID = os.getenv('RUN_ID') # <-- AWS

# logged_model = f's3://mlflow-models-alexey/1/{RUN_ID}/artifacts/model' # <-- AWS
logged_model = f'runs:/{RUN_ID}/model' # <-- Local
model = mlflow.pyfunc.load_model(logged_model)


def prepare_features(ride: dict[str,float]) -> dict[str,float|str]:
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features


def predict(features: dict[str,float|str]) -> float:
    preds = model.predict(features)
    return float(preds[0])


app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred,
        'model_version': RUN_ID
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
