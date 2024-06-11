import io
import json

from flask import Flask, jsonify, make_response, render_template, request
from PIL import Image
from predict import predict_disease

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    json_response = {}
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


if __name__ == "__main__":
    app.run(debug=True, port=3000)
