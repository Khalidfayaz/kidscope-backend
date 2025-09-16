from flask import Flask
from ocr import app as ocr_app
from report import app as report_app

app = Flask(__name__)

# Register OCR routes under /ocr/*
app.register_blueprint(ocr_app.blueprints[None], url_prefix="/ocr")

# Register Report routes under /report/*
app.register_blueprint(report_app.blueprints[None], url_prefix="/report")

@app.route("/")
def home():
    return {"message": "Backend Running - OCR & Report services"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
