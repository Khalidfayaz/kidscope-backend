from flask import Flask
from ocr import ocr_bp
from report import report_bp

app = Flask(__name__)

# Register routes
app.register_blueprint(ocr_bp, url_prefix="/ocr")
app.register_blueprint(report_bp, url_prefix="/report")

@app.route("/")
def home():
    return {"message": "Kidscope Backend Running 🎉"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
