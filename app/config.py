# config.py
import os


class Config:
    # Flask configuration
    SECRET_KEY = 'dermai'

    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URI", 'postgresql://group5:dermai@localhost/dermai')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
