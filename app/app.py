import io
import json

from flask import Flask, jsonify, make_response, render_template, request
from MyNet import MyNet
from PIL import Image
from predict import predict_disease
from progress_tracking import predict_acne_severity

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signin')
def signin():
    return render_template('signin.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        try:
            file = request.files['file']
        except KeyError:
            return make_response(jsonify({
                'error': 'No file part in the request',
                'code': 'FILE',
                'message': 'file is not valid'
            }), 400)

        image_pil = Image.open(io.BytesIO(file.read()))
        image_bytes_io = io.BytesIO()
        image_pil.save(image_bytes_io, format='JPEG')
        image_bytes_io.seek(0)
        path = image_bytes_io

        response = predict_disease(path)
        response["img_path"] = file.filename

        print(f"Final response: {response}")
        return make_response(jsonify(response), 200)
    else:
        return render_template('detect.html')


@app.route('/progress', methods=['GET', 'POST'])
def progress_tracking():
    if request.method == 'POST':
        try:
            file = request.files['file']
        except KeyError:
            return make_response(jsonify({
                'error': 'No file part in the request',
                'code': 'FILE',
                'message': 'file is not valid'
            }), 400)

        image_pil = Image.open(io.BytesIO(file.read()))
        image_bytes_io = io.BytesIO()
        image_pil.save(image_bytes_io, format='JPEG')
        image_bytes_io.seek(0)
        path = image_bytes_io

        response = predict_acne_severity(path)
        response["img_path"] = file.filename

        print(f"Final response: {response}")
        return make_response(jsonify(response), 200)
    else:
        return render_template('detect.html')

if __name__ == "__main__":
    app.run(debug=True, port=3000)

