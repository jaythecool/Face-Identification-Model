from flask import Flask, request, jsonify
import util

app = Flask(__name__)

@app.route("/classify_image", methods = ["get", "post"])
def classify_image():
    img_data = request.form["img_data"]

    response = jsonify(util.classifyImg(img_data))
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == "__main__":
    print("Starting python flask server to cricketers identification")
    util.load_artifacts()
    app.run(port=5000)