from flask import Flask, request, jsonify
import json
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/update_model", methods=["POST"])
def update_model():
    print("Request received")
    data = request.json
    print(f"Data: {data}")
    config_path = os.path.join(os.path.dirname(__file__), "config.json")

    # Read the existing config or initialize an empty one
    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            config = json.load(file)
    else:
        config = {}

    # Update the config with new values from the request
    config.update(data)

    # Write the updated config back to the file
    with open(config_path, "w") as file:
        json.dump(config, file, indent=4)

    return jsonify({"message": "Model updated successfully"}), 200


if __name__ == "__main__":
    app.run(debug=True)
