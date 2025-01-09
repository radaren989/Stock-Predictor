from flask import Flask
from routes import app_routes  # 'main' yerine 'app_routes' import ediyoruz

app = Flask(__name__)

# Blueprint'i kaydediyoruz
app.register_blueprint(app_routes)

if __name__ == '__main__':
    app.run(debug=True)
