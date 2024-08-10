# config.py
import os


class Config:
    # Flask configuration
    SECRET_KEY = os.environ.get("APP_SECRET")
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URI")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
