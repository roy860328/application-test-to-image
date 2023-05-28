import os, sys
from flask import Flask, request, jsonify, render_template
from io import BytesIO
import base64

from models.backend_model import inference

# flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/image", methods=["POST"])
def generate_image_endpoint():
    data = request.get_json()
    text = data["text"]
    path_img = inference(text)
    return jsonify({"image": path_img})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
