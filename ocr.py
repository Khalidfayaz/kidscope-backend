from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import json
import re
import pandas as pd
import tempfile
from io import BytesIO
from PIL import Image
from pdf2image import convert_from_path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# =======================
# CONFIGURATION
# =======================
GRADE_TO_MARKS = {
    "A+": 95,
    "A": 85,
    "B+": 75,
    "B": 65,
    "C+": 55,
    "C": 45,
    "D+": 35,
    "D": 25,
    "E": 15,
    "F": 0
}

def gpa_to_grade(gpa):
    """Convert GPA to letter grade"""
    if gpa is None or gpa == "N/A":
        return "N/A"
    try:
        gpa = float(gpa)
        if gpa >= 3.6:
            return "A"
        elif gpa >= 3.2:
            return "B+"
        elif gpa >= 2.8:
            return "B"
        elif gpa >= 2.4:
            return "C+"
        elif gpa >= 2.0:
            return "C"
        elif gpa >= 1.6:
            return "D+"
        elif gpa >= 1.2:
            return "D"
        else:
            return "E"
    except:
        return "N/A"

# =======================
# UTILITIES
# =======================
def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def clean_json_response(response_text):
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group()
        try:
            return json.loads(json_str)
        except:
            try:
                return json.loads(json_str.replace("\n", " ").replace("\t", " "))
            except:
                return None
    return None

def extract_data_from_image(image_base64):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are an expert in extracting Indian school marksheets (CBSE, ICSE, state boards, private schools).
Always output JSON with:
student_name, father_name, mother_name, roll_number, registration_number, class_grade, section, academic_year/session, exam_name, school_name,
subjects (subject_name, theory_marks, practical_marks, total_marks, grade),
obtained_marks, total_marks, percentage, overall_grade, pass_fail, remarks, other_info (include GPA if present).
"""
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                    }
                ]
            }
        ],
        max_tokens=2000,
        temperature=0.1,
    )
    return response.choices[0].message.content

def extract_data_from_excel(file_content, filename):
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(BytesIO(file_content))
        else:
            df = pd.read_excel(BytesIO(file_content))

        subjects = []
        total_obtained = 0
        total_marks = 0

        for _, row in df.iterrows():
            subject_name = str(row.get("Subject", "N/A"))
            obtained = row.get("Marks Obtained", "N/A")
            maximum = row.get("Total Marks", 100)

            # Convert grade → marks
            if isinstance(obtained, str) and obtained in GRADE_TO_MARKS:
                obtained = GRADE_TO_MARKS[obtained]

            try:
                maximum = int(maximum)
            except:
                maximum = 100

            if obtained == "N/A":
                obtained = 0

            subjects.append({
                "subject_name": subject_name,
                "theory_marks": int(obtained),
                "total_marks": maximum,
                "grade": "N/A"
            })
            total_obtained += int(obtained)
            total_marks += maximum

        percentage = round((total_obtained / total_marks) * 100, 2) if total_marks else "N/A"

        return {
            "student_name": "N/A",
            "roll_number": "N/A",
            "registration_number": "N/A",
            "class_grade": "N/A",
            "section": "N/A",
            "academic_year": "N/A",
            "exam_name": "N/A",
            "school_name": "N/A",
            "subjects": subjects,
            "obtained_marks": total_obtained,
            "total_marks": total_marks,
            "percentage": percentage,
            "overall_grade": "N/A",
            "pass_fail": "Pass" if percentage != "N/A" and percentage >= 40 else "Fail",
            "remarks": "N/A",
            "other_info": "Extracted from Excel/CSV"
        }
    except Exception as e:
        return {"error": f"Excel parsing error: {str(e)}"}

# =======================
# API ROUTES
# =======================
@app.route("/api/extract-marksheet", methods=["POST"])
def extract_marksheet():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename
    if not filename:
        return jsonify({"error": "No file selected"}), 400

    file_content = file.read()
    results = []

    # Handle Excel/CSV
    if filename.lower().endswith((".xls", ".xlsx", ".csv")):
        data = extract_data_from_excel(file_content, filename)
        results.append({"page": 1, "data": data})
        return jsonify({"filename": filename, "pages_processed": 1, "results": results})

    # Handle PDF/Image
    images = []
    if filename.lower().endswith(".pdf"):
        # ✅ Cross-platform PDF handling
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_content)
            temp_path = tmp.name

        try:
            images = convert_from_path(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    else:
        try:
            images.append(Image.open(BytesIO(file_content)))
        except Exception as e:
            return jsonify({"error": f"Image error: {str(e)}"}), 500

    for idx, image in enumerate(images):
        img_b64 = encode_image(image)
        raw_data = extract_data_from_image(img_b64)
        if raw_data:
            cleaned = clean_json_response(raw_data)
            if cleaned:
                # Normalize subjects
                total_obtained = 0
                total_marks = 0
                normalized_subjects = []

                for subj in cleaned.get("subjects", []):
                    grade = subj.get("grade", "N/A")
                    theory = subj.get("theory_marks", "N/A")
                    maximum = subj.get("total_marks", 100)

                    # Convert grade → marks
                    if isinstance(theory, str) and theory in GRADE_TO_MARKS:
                        theory = GRADE_TO_MARKS[theory]
                    elif theory in ["N/A", None, ""]:
                        if grade in GRADE_TO_MARKS:
                            theory = GRADE_TO_MARKS[grade]
                        else:
                            theory = 0

                    try:
                        maximum = int(maximum)
                    except:
                        maximum = 100

                    obtained_val = int(theory) if isinstance(theory, (int, float)) else 0

                    normalized_subjects.append({
                        "subject_name": subj.get("subject_name", "Unknown"),
                        "marks_obtained": obtained_val,   # ✅ added
                        "theory_marks": obtained_val,
                        "total_marks": maximum,
                        "grade": grade
                    })

                    total_obtained += obtained_val
                    total_marks += maximum

                # Replace subjects with normalized version
                cleaned["subjects"] = normalized_subjects

                # Totals
                if total_marks > 0:
                    cleaned["obtained_marks"] = total_obtained
                    cleaned["total_marks"] = total_marks
                    cleaned["percentage"] = round((total_obtained / total_marks) * 100, 2)
                    cleaned["pass_fail"] = "Pass" if cleaned["percentage"] >= 40 else "Fail"

                # GPA → Grade mapping
                other_info = cleaned.get("other_info", "")
                gpa_value = None
                if isinstance(other_info, dict) and "GPA" in other_info:
                    gpa_value = other_info["GPA"]
                elif isinstance(other_info, str) and "GPA" in other_info:
                    gpa_value = other_info.split(":")[-1].strip()

                if gpa_value:
                    cleaned["overall_grade"] = gpa_to_grade(gpa_value)

                results.append({"page": idx + 1, "data": cleaned})

                # After cleaning subjects and totals
                if cleaned:
                    source_type = "marks"

                # Detect grade-based
                if any(sub.get("grade") not in [None, "", "N/A"] for sub in cleaned.get("subjects", [])):
                    source_type = "grade"

                # Detect GPA-based
                if isinstance(cleaned.get("other_info"), dict) and "GPA" in cleaned["other_info"]:
                    source_type = "gpa"

                cleaned["source_type"] = source_type

                # Normalize session field
                session_val = cleaned.get("academic_year") or cleaned.get("session") or None
                if session_val:
                    cleaned["session"] = str(session_val)



    return jsonify({"filename": filename, "pages_processed": len(images), "results": results})

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "marksheet-extractor"})

# =======================
# MAIN
# =======================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
