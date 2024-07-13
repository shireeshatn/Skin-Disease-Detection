# config.py

class Config:
    # Flask configuration
    SECRET_KEY = 'dermai'
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = 'postgresql://group5:dermai@localhost/dermai'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
