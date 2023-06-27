import pickle

from flask import Flask, request, jsonify

# Load the pickled model saved in Week 1
with open('lin_reg.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)


def prepare_features(ride: dict[str,float]) -> dict[str,float|str]:
    # We will save the features in a dictionary
    features = {}
    # Preprocessing
    features['PU_DO'] = f'{ride["PULocationID"]}_{ride["DOLocationID"]}'
    features['trip_distance'] = ride['trip_distance']
    return features


# Predicting from features given as dict
def predict(features: dict[str,float|str]) -> float:
    X = dv.transform(features)
    preds = model.predict(X)  # returning numpy array
    return float(preds[0])    # Only returns 1st prediction


app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint(): # <-- The parameters are usually given by Flask

    ride = request.get_json()  # <-- The parameters are extracted from the
                               #     request (reads the JSON passed to the app)

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred
    }

    return jsonify(result) # transforms a dictionary into a JSON


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)