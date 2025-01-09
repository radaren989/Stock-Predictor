from flask import Flask
from routes import app_routes  # routes dosyasındaki app_routes'u import ediyoruz


def create_app():
    app = Flask(__name__)

    # Blueprint'i kaydediyoruz
    app.register_blueprint(app_routes)

    return app
