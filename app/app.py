import os
from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

load_dotenv()

# Initialize Flask application
app = Flask(__name__)
app.config.from_object('config.Config')

# Initialize the database
db = SQLAlchemy(app)

# MyNet class for acne progress tracking model
import torch.nn as nn
import torchvision


# Database models
class User(db.Model):
    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    is_guest = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return '<User %r>' % self.username

class AcneProgress(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), nullable=False)  # Corrected foreign key reference
    acne_level = db.Column(db.Integer, nullable=False)
    date_recorded = db.Column(db.DateTime, nullable=False)

    user = db.relationship('User', backref=db.backref('acne_progress', lazy=True))  # Optional: add relationship for easy access


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.cnn = torchvision.models.efficientnet_v2_m(pretrained=True).cuda()
        for param in self.cnn.parameters():
            param.requires_grad = True
        self.cnn.classifier = nn.Sequential(
            nn.Linear(self.cnn.classifier[1].in_features, 512),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.Linear(64, 4),
        )

    def forward(self, img):
        output = self.cnn(img)
        return output

# Function to predict acne severity
from acne_severity_classifier import predict_acne_severity
# Function to load and preprocess the image for skin disease detection
from skin_disease_detection import predict_disease

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}  # Allowed file extensions

# Function to check if the filename has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes for the application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_login', methods=['GET'])
def check_login():
    if 'logged_in' in session:
        return {'logged_in': session['logged_in']}
    else:
        return {'logged_in': False}

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'logged_in' in session and session['logged_in']:
        return render_template('index.html')

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.user_id
            session['logged_in'] = True
            return redirect(url_for('index'))

        session['logged_in'] = False
        return render_template('login.html', error='Invalid username or password.')

    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password)

        db.session.add(new_user)
        db.session.commit()

        # Check if login successful message should be displayed

        return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session['logged_in'] = False
    return redirect(url_for('index'))


@app.route('/detect_disease', methods=['POST'])
def detect_disease():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        # Save the file to a temporary location
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.root_path, 'static', 'uploads', filename)
        file.save(file_path)

        # Perform skin disease detection
        result = predict_disease(file_path)

        # Remove the temporary file
        os.remove(file_path)
        return jsonify({'disease_names': result['diseases'], 'probabilities': result['predictions']})
    else:
        return jsonify({'error': 'File type not allowed'})


@app.route('/track_acne', methods=['POST'])
def track_acne():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']

    if file.filename == '':
        return redirect(request.url)

    # Save the file to a temporary location
    file_path = os.path.join(app.root_path, 'static', 'uploads', file.filename)
    file.save(file_path)

    # Perform acne severity prediction
    result = predict_acne_severity(file_path)

    # Record acne progress if user logged in
    if 'user_id' in session:
        user_id = session['user_id']
        acne_level = result['predicted_class']  # Adjust as per model output

        # Save to database
        new_progress = AcneProgress(user_id=user_id, acne_level=acne_level, date_recorded=datetime.now())
        print(new_progress)
        db.session.add(new_progress)
        db.session.commit()

    # Remove the temporary file
    os.remove(file_path)

    return jsonify({'current_level': result['severity']})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=3000)
