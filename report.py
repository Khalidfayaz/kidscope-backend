from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from datetime import datetime
import time
import json
import hashlib
import textwrap
from pathlib import Path
from typing import List, Tuple, Optional

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA

# -------------------- Flask App & CORS --------------------
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://oohr-erp.web.app",
    ]
    if origin in allowed_origins:
        response.headers.add('Access-Control-Allow-Origin', origin)
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# -------------------- Environment --------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# -------------------- AI Components --------------------
FAISS_INDEX_PATH = "faiss_index"
embeddings = OpenAIEmbeddings()
vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
llm = ChatOpenAI(model="gpt-4o", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)

# -------------------- Zodiac --------------------
ZODIAC_SIGNS = [
    ("Capricorn", (1, 1), (1, 19)),
    ("Aquarius", (1, 20), (2, 18)),
    ("Pisces", (2, 19), (3, 20)),
    ("Aries", (3, 21), (4, 19)),
    ("Taurus", (4, 20), (5, 20)),
    ("Gemini", (5, 21), (6, 20)),
    ("Cancer", (6, 21), (7, 22)),
    ("Leo", (7, 23), (8, 22)),
    ("Virgo", (8, 23), (9, 22)),
    ("Libra", (9, 23), (10, 22)),
    ("Scorpio", (10, 23), (11, 21)),
    ("Sagittarius", (11, 22), (12, 21)),
    ("Capricorn", (12, 22), (12, 31)),
]

FAMOUS_ZODIACS = {
    "Aries": ["Ajay Devgn", "Kapil Sharma", "Dr. A.P.J. Abdul Kalam", "Emraan Hashmi", "Robert Downey Jr."],
    "Taurus": ["Sachin Tendulkar", "Anushka Sharma", "G. D. Naidu", "Madhuri Dixit", "David Beckham"],
    "Gemini": ["Sonam Kapoor", "Shilpa Shetty", "Karan Johar", "Dr. B. R. Ambedkar", "Angelina Jolie"],
    "Cancer": ["Priyanka Chopra", "MS Dhoni", "Ranveer Singh", "J. R. D. Tata", "Ariana Grande"],
    "Leo": ["Saif Ali Khan", "Sridevi", "Jacqueline Fernandez", "Bal Gangadhar Tilak", "Barack Obama"],
    "Virgo": ["Akshay Kumar", "Kareena Kapoor", "Narendra Modi", "Verghese Kurien", "Michael Jackson"],
    "Libra": ["Amitabh Bachchan", "Rekha", "Ranbir Kapoor", "Dr. Vikram Sarabhai", "Will Smith"],
    "Scorpio": ["Shah Rukh Khan", "Aishwarya Rai", "Sushmita Sen", "Lal Bahadur Shastri", "Bill Gates"],
    "Sagittarius": ["Yami Gautam", "Dharmendra", "John Abraham", "Kalpana Chawla", "Taylor Swift"],
    "Capricorn": ["Deepika Padukone", "Hrithik Roshan", "Javed Akhtar", "Swami Vivekananda", "Michelle Obama"],
    "Aquarius": ["Preity Zinta", "Abhishek Bachchan", "Jackie Shroff", "Ratan Tata", "Oprah Winfrey"],
    "Pisces": ["Alia Bhatt", "Shahid Kapoor", "Tiger Shroff", "C. V. Raman", "Albert Einstein"]
}

def get_zodiac_and_famous_people(dob_str):
    try:
        dob = datetime.strptime(dob_str, "%Y-%m-%d")
        month, day = dob.month, dob.day
        for sign, start, end in ZODIAC_SIGNS:
            if (month, day) >= start and (month, day) <= end:
                return sign, FAMOUS_ZODIACS.get(sign, [])
    except Exception as e:
        print("Zodiac parsing error:", e)
    return "Unknown", []

# -------------------- Helpers --------------------
def format_response_item(item):
    if not isinstance(item, str):
        return item
    key_terms = [
        "Social Engagement", "Self-Efficacy", "Temperament", 
        "Internalizing", "Self-Esteem", "School Refusal",
        "Emotional Expression", "Dependent Behavior", 
        "Parental Reinforcement", "Communication",
        "Independence", "Social Interaction"
    ]
    for term in key_terms:
        if term in item and f"**{term}**" not in item:
            item = item.replace(term, f"**{term}**")
    return item

def parse_report_sections(text):
    sections = {"strengths": [], "weaknesses": [], "recommendations": []}
    current_section = None

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        lower = line.lower()

        # Detect headings (more flexible)
        if "strength" in lower:
            current_section = "strengths"
            continue
        if "weakness" in lower or "area for improvement" in lower or "areas for improvement" in lower:
            current_section = "weaknesses"
            continue
        if "recommendation" in lower:
            current_section = "recommendations"
            continue

        # Add content
        if current_section:
            sections[current_section].append(format_response_item(line))

    # Limit to 3 items per section, fallback if empty
    for k in sections:
        sections[k] = sections[k][:3]
        if not sections[k]:
            sections[k] = [f"No {k} identified"]

    return sections


# -------------------- Discussion Bot Functions --------------------
def week_seed():
    """Return current ISO calendar week as string e.g. '2025-W36'"""
    today = datetime.now().date()
    y, w, _ = today.isocalendar()
    return f"{y}-W{w}"

def generate_discussion_questions(report_data, personal_info):
    """Generate short discussion questions based on the complete report and personal info"""
    try:
        # Create a focused summary of the report for context
        report_summary = f"""
        CHILD PROFILE:
        - Name: {personal_info.get('name', 'Unknown')}
        - Age: {personal_info.get('age', 'Unknown')}
        - Date of Birth: {personal_info.get('dob', 'Unknown')}
        - Zodiac Sign: {report_data.get('zodiac', 'Unknown')}
        
        Key Strengths (first 2):
        {chr(10).join(['• ' + strength for strength in report_data.get('strengths', [])[:2]])}
        
        Areas for Improvement (first 2):
        {chr(10).join(['• ' + weakness for weakness in report_data.get('weaknesses', [])[:2]])}
        """
        
        # Construct query for short question generation
        query = f"""
        Generate 3-5 very short, simple questions for a child based on their psychological profile.
        
        Profile Summary:
        {report_summary}
        
        GUIDELINES:
        1. Keep questions VERY SHORT (max 10-12 words each)
        2. Focus on one specific aspect of their report
        3. Make questions simple and easy to understand
        4. Use child-friendly language
        5. Reference their specific strengths or challenges
        6. Questions should be open-ended but concise
        
        Return ONLY the questions as a numbered list, nothing else.
        """
        
        # Call QA chain to generate questions
        result = qa_chain({"query": query})
        questions_text = result.get("result", "")
        
        # Parse the response to extract questions
        questions = []
        lines = questions_text.split('\n')
        for line in lines:
            line = line.strip()
            # Extract numbered questions (1., 2., etc.)
            if line and any(line.startswith(f"{i}.") for i in range(1, 10)):
                # Remove the number and dot
                question = line.split('.', 1)[1].strip() if '. ' in line else line
                if question and len(question) > 5:  # Ensure it's a meaningful question
                    questions.append(question)
            # Extract bulleted questions
            elif line and (line.startswith('-') or line.startswith('•')):
                question = line[1:].strip()
                if question and len(question) > 5:
                    questions.append(question)
        
        # If we couldn't parse questions properly, use fallback
        if not questions or len(questions) < 3:
            questions = get_short_fallback_questions(report_data, personal_info)
            
        return questions[:5]  # Return up to 5 questions
        
    except Exception as e:
        print("Error generating discussion questions:", e)
        return get_short_fallback_questions(report_data, personal_info)

def get_short_fallback_questions(report_data, personal_info):
    """Fallback short questions"""
    name = personal_info.get('name', 'there')
    
    # Get first strength and weakness
    strengths = report_data.get('strengths', [])
    weaknesses = report_data.get('weaknesses', [])
    
    strength_text = ""
    if strengths:
        clean_strength = strengths[0].replace('**', '').strip().lower()
        strength_text = f"your {clean_strength}"
    
    weakness_text = ""
    if weaknesses:
        clean_weakness = weaknesses[0].replace('**', '').strip().lower()
        weakness_text = f"about {clean_weakness}"
    
    base_questions = []
    
    if strength_text:
        base_questions.append(f"What do you enjoy about {strength_text}?")
        base_questions.append(f"How does {strength_text} make you feel?")
    
    if weakness_text:
        base_questions.append(f"How do you feel about {weakness_text}?")
        base_questions.append(f"What helps you with {weakness_text}?")
    
    # Add general questions if we need more
    general_questions = [
        f"What makes you happy, {name}?",
        "What's your favorite thing to do?",
        "Who do you like spending time with?",
        "What are you good at?"
    ]
    
    # Combine and return
    all_questions = base_questions + general_questions
    return all_questions[:5]  # Return up to 5 questions


# -------------------- Routes --------------------
@app.route('/rag', methods=['OPTIONS'])
def handle_options():
    return jsonify({'message': 'Preflight request accepted'}), 200

@app.route("/rag", methods=["POST"])
def rag():
    try:
        data = request.get_json()
        print("\n📩 [REQUEST RECEIVED - /rag]:", data, flush=True)  # log request

        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # Required fields
        for field in ['dob', 'time_of_birth', 'place_of_birth', 'symptom_keywords']:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        dob = data['dob']
        time_of_birth = data['time_of_birth']
        place_of_birth = data['place_of_birth']

        # Convert symptom_keywords to list if it's an object
        symptoms = data['symptom_keywords']
        if isinstance(symptoms, dict):
            symptoms = list(symptoms.values())
        elif not isinstance(symptoms, list):
            symptoms = []

        academic_records = data.get('academic_records', [])

        # Zodiac
        zodiac, famous_people = get_zodiac_and_famous_people(dob)

        # Academic summary
        academic_summary = ""
        if isinstance(academic_records, str):
            academic_summary = f"\nAcademic Performance:\n{academic_records}"
        elif isinstance(academic_records, list):
            academic_summary = "\nAcademic Performance:\n" + "\n".join(
                f"{rec.get('year','')} - Class {rec.get('class','')}: " +
                ", ".join(f"{sub['subject']} ({sub['percentage']}%)" 
                for sub in rec.get('subjects',[]))
                for rec in academic_records
            )

        # Construct query
        query = f"""
Comprehensive Child Profile Analysis Request:

🧠 Basic Information:
- Date of Birth: {dob}
- Time of Birth: {time_of_birth}
- Place of Birth: {place_of_birth}
- Zodiac Sign: {zodiac}
- Famous People with Same Sign: {', '.join(famous_people)}

🧩 Psychological Traits (DSM-5 indicators):
{', '.join(symptoms)}

📘 Academic Performance Summary:
{academic_summary if academic_summary else "Academic records were not provided."}

📊 Please provide:
1. Three Key Strengths
2. Three Areas for Improvement
3. Three Personalized Recommendations

💡 Notes:
- Bold important traits (**like this**)
"""

        # Call QA chain
        result = qa_chain({"query": query})
        full_answer = result.get("result", "No AI response generated.")

        sections = parse_report_sections(full_answer)

        response_data = {
            "strengths": sections["strengths"],
            "weaknesses": sections["weaknesses"],
            "recommendations": sections["recommendations"],
            "zodiac": zodiac,
            "famous_people": famous_people,
            "raw_answer": full_answer
        }

        print("📤 [RESPONSE SENT - /rag]:", response_data, flush=True)  # log response
        return jsonify(response_data)

    except Exception as e:
        print("Error in /rag endpoint:", e, flush=True)
        return jsonify({"error": "Failed to generate report", "details": str(e)}), 500

# 🔹 NEW ROUTE: Generate discussion questions based on report and personal info
@app.route("/discussion-questions", methods=["POST"])
def discussion_questions():
    try:
        data = request.get_json()
        print("\n📩 [REQUEST RECEIVED - /discussion-questions]:", data, flush=True)

        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        report_data = data.get("report", {})
        personal_info = data.get("personal_info", {})
        
        if not report_data:
            return jsonify({"error": "Missing required field: report"}), 400

        questions = generate_discussion_questions(report_data, personal_info)
        
        response_data = {
            "questions": questions,
            "week_tag": week_seed()
        }

        print("📤 [RESPONSE SENT - /discussion-questions]:", response_data, flush=True)
        return jsonify(response_data)

    except Exception as e:
        print("Error in /discussion-questions endpoint:", e, flush=True)
        return jsonify({"error": "Failed to generate discussion questions", "details": str(e)}), 500

# In report.py, update the discussion endpoints for better responses

@app.route("/discussion-followup", methods=["POST"])
def discussion_followup():
    try:
        data = request.get_json()
        clicked_question = data.get("question", "")
        report_data = data.get("report", {})
        personal_info = data.get("personal_info", {})
        conversation_history = data.get("conversation_history", [])

        # Build conversation context
        history_context = ""
        if conversation_history:
            history_context = "Previous conversation:\n"
            for msg in conversation_history[-4:]:  # Last 4 messages for context
                role = "Child" if msg.get("role") == "user" else "Assistant"
                history_context += f"{role}: {msg.get('text', '')}\n"

        query = f"""
        You are a compassionate child psychologist named Saarthi having a conversation with a child.
        
        {history_context}
        
        The child was asked this question: "{clicked_question}".

        CHILD'S COMPLETE PROFILE:
        - Name: {personal_info.get('name', 'Unknown')}
        - Age: {personal_info.get('age', 'Unknown')}
        - Date of Birth: {personal_info.get('dob', 'Unknown')}
        - Zodiac Sign: {report_data.get('zodiac', 'Unknown')}
        
        PSYCHOLOGICAL REPORT:
        Strengths: {report_data.get('strengths', [])}
        Areas for Improvement: {report_data.get('weaknesses', [])}
        Recommendations: {report_data.get('recommendations', [])}
        
        TASK:
        1. Provide a detailed, empathetic response to the child's answer (3-4 sentences)
        2. Reference specific aspects of their psychological profile where relevant
        3. Use a warm, supportive, and encouraging tone
        4. Make it personal by using their name and specific details from their report
        5. Then suggest 3 follow-up questions that continue the conversation naturally

        RESPONSE FORMAT:
        {{
          "answer": "Your detailed empathetic response here (3-4 sentences)",
          "questions": ["Follow-up question 1", "Follow-up question 2", "Follow-up question 3"]
        }}
        """

        result = qa_chain({"query": query})
        reply_text = result.get("result", "")

        # --- Clean JSON safely ---
        try:
            cleaned = reply_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned.replace("```json", "").replace("```", "").strip()
            elif "{" in cleaned and "}" in cleaned:
                # Extract JSON from text that might have other content
                json_start = cleaned.find("{")
                json_end = cleaned.rfind("}") + 1
                cleaned = cleaned[json_start:json_end]

            parsed = json.loads(cleaned)
            answer = parsed.get("answer", "That's really interesting! 😊 I'd love to hear more about that.")
            questions = parsed.get("questions", [])
        except Exception as e:
            print("JSON parsing error, using fallback:", e)
            # Try to extract answer and questions from text
            lines = [l.strip() for l in reply_text.split("\n") if l.strip()]
            answer = ""
            questions = []
            
            # Find the answer section
            answer_keywords = ["answer", "response", ":"]
            for i, line in enumerate(lines):
                if any(keyword in line.lower() for keyword in answer_keywords) and "question" not in line.lower():
                    answer = line.split(":", 1)[1].strip() if ":" in line else line
                    break
            
            if not answer and lines:
                answer = lines[0]
            
            # Find questions
            for line in lines:
                if (line.startswith("-") or line.startswith("•") or line[0].isdigit() or 
                    "question" in line.lower()) and len(line) > 10:
                    clean_line = line[1:].strip() if line.startswith("-") or line.startswith("•") else line
                    clean_line = clean_line.split(".", 1)[1].strip() if ". " in clean_line else clean_line
                    if "question" not in clean_line.lower() and len(clean_line) > 10:
                        questions.append(clean_line)
            
            if not answer:
                answer = "That's really interesting! 😊 I'd love to hear more about that."
            
            if len(questions) < 3:
                questions.extend([
                    "Can you tell me more about that?",
                    "What made you feel that way?",
                    "How would you like to build on this strength?"
                ])

        # Ensure we have exactly 3 questions
        questions = questions[:3]
        while len(questions) < 3:
            questions.append("What are your thoughts about that?")

        return jsonify({"answer": answer, "questions": questions})

    except Exception as e:
        print("Error in /discussion-followup:", e)
        return jsonify({
            "answer": "That's really interesting! 😊 Thanks for sharing. I'd love to hear more about your experiences and thoughts.",
            "questions": [
                "Can you tell me more about that?",
                "What made you feel that way?",
                "How would you like to build on this?"
            ]
        }), 200

@app.route("/discussion-free", methods=["POST"])
def discussion_free():
    try:
        data = request.get_json()
        user_question = data.get("question", "")
        report_data = data.get("report", {})
        personal_info = data.get("personal_info", {})
        conversation_history = data.get("conversation_history", [])

        # Build conversation context
        history_context = ""
        if conversation_history:
            history_context = "Previous conversation:\n"
            for msg in conversation_history[-4:]:  # Last 4 messages for context
                role = "Child" if msg.get("role") == "user" else "Assistant"
                history_context += f"{role}: {msg.get('text', '')}\n"

        query = f"""
        You are a compassionate child psychologist named Saarthi having a conversation with a child.
        
        {history_context}
        
        The child asked: "{user_question}".

        CHILD'S COMPLETE PROFILE:
        - Name: {personal_info.get('name', 'Unknown')}
        - Age: {personal_info.get('age', 'Unknown')}
        - Date of Birth: {personal_info.get('dob', 'Unknown')}
        - Zodiac Sign: {report_data.get('zodiac', 'Unknown')}
        
        PSYCHOLOGICAL REPORT:
        Strengths: {report_data.get('strengths', [])}
        Areas for Improvement: {report_data.get('weaknesses', [])}
        Recommendations: {report_data.get('recommendations', [])}
        
        TASK:
        1. Provide a detailed, empathetic response to the child's question (3-4 sentences)
        2. Reference specific aspects of their psychological profile where relevant
        3. Use a warm, supportive, and encouraging tone
        4. Make it personal by using their name and specific details from their report
        5. Then suggest 3 follow-up questions that continue the conversation naturally

        RESPONSE FORMAT:
        {{
          "answer": "Your detailed empathetic response here (3-4 sentences)",
          "questions": ["Follow-up question 1", "Follow-up question 2", "Follow-up question 3"]
        }}
        """

        result = qa_chain({"query": query})
        reply_text = result.get("result", "")

        # --- Clean JSON safely ---
        try:
            cleaned = reply_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned.replace("```json", "").replace("```", "").strip()
            elif "{" in cleaned and "}" in cleaned:
                # Extract JSON from text that might have other content
                json_start = cleaned.find("{")
                json_end = cleaned.rfind("}") + 1
                cleaned = cleaned[json_start:json_end]

            parsed = json.loads(cleaned)
            answer = parsed.get("answer", "That's a great question! 😊 Let's explore that together.")
            questions = parsed.get("questions", [])
        except Exception as e:
            print("JSON parsing error, using fallback:", e)
            # Try to extract answer and questions from text
            lines = [l.strip() for l in reply_text.split("\n") if l.strip()]
            answer = ""
            questions = []
            
            # Find the answer section
            answer_keywords = ["answer", "response", ":"]
            for i, line in enumerate(lines):
                if any(keyword in line.lower() for keyword in answer_keywords) and "question" not in line.lower():
                    answer = line.split(":", 1)[1].strip() if ":" in line else line
                    break
            
            if not answer and lines:
                answer = lines[0]
            
            # Find questions
            for line in lines:
                if (line.startswith("-") or line.startswith("•") or line[0].isdigit() or 
                    "question" in line.lower()) and len(line) > 10:
                    clean_line = line[1:].strip() if line.startswith("-") or line.startswith("•") else line
                    clean_line = clean_line.split(".", 1)[1].strip() if ". " in clean_line else clean_line
                    if "question" not in clean_line.lower() and len(clean_line) > 10:
                        questions.append(clean_line)
            
            if not answer:
                answer = "That's a great question! 😊 Let's explore that together."
            
            if len(questions) < 3:
                questions.extend([
                    "Can you tell me more about what you're thinking?",
                    "What made you curious about this?",
                    "How do you feel about this topic?"
                ])

        # Ensure we have exactly 3 questions
        questions = questions[:3]
        while len(questions) < 3:
            questions.append("What are your thoughts about that?")

        return jsonify({"answer": answer, "questions": questions})

    except Exception as e:
        print("Error in /discussion-free:", e)
        return jsonify({
            "answer": "That's a really good question! 😊 I'd love to explore that with you. Your curiosity is wonderful!",
            "questions": [
                "Can you tell me more about what you're thinking?",
                "What made you curious about this?",
                "How do you feel about this topic?"
            ]
        }), 200


# -------------------- Main --------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
