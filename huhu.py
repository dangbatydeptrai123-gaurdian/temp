import os, re, json, math, csv
from datetime import datetime
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict

from flask import Flask, jsonify, request
from flask_cors import CORS
from google import genai
from google.genai import types as gtypes
from unidecode import unidecode

from flask_sqlalchemy import SQLAlchemy

# =========================
# Existing app & datastore
# =========================
app = Flask(__name__)
CORS(app)

# Configure database (SQLite)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///messages.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Model ---
class UserMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.String(50), unique=True)
    messages = db.Column(db.Text)  # stores all messages as one long text

    def to_dict(self):
        return {"sender_id": self.sender_id, "messages": self.messages}

# Create table if not exist
with app.app_context():
    db.create_all()

# --- Helper: extract id + text (kept as-is) ---
def extract_message(data):
    entry = data[0]["body"]["entry"][0]
    msg = entry["messaging"][0]
    return msg["sender"]["id"], msg["message"]["text"]

# --- Existing routes (kept) ---
@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()
    try:
        sender_id, text = extract_message(data)

        user = UserMessage.query.filter_by(sender_id=sender_id).first()
        if user:
            # append new text
            user.messages = (user.messages or "") + f"\n{text}"
        else:
            # create new sender record
            user = UserMessage(sender_id=sender_id, messages=text)
            db.session.add(user)

        db.session.commit()
        return jsonify({"message": "Saved successfully"}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/messages', methods=['GET'])
def get_all():
    users = UserMessage.query.all()
    return jsonify([u.to_dict() for u in users])

# =========================
# RAG Utilities (no numpy)
# =========================

_WORD_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)

def normalize_text(s: str) -> str:
    # Lowercase, strip accents
    return unidecode((s or "").lower())

def tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    return _WORD_RE.findall(s)

def split_into_chunks(text: str, max_chars: int = 800, overlap: int = 100) -> List[str]:
    """
    Simple character-based chunking with overlap; preserves line boundaries if possible.
    """
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    pieces: List[str] = []
    buf = ""
    for ln in lines:
        if len(buf) + len(ln) + 1 <= max_chars:
            buf = (buf + "\n" + ln) if buf else ln
        else:
            if buf:
                pieces.append(buf)
            # start new buffer; if line is very long, force-slice
            if len(ln) <= max_chars:
                buf = ln
            else:
                start = 0
                while start < len(ln):
                    end = min(start + max_chars, len(ln))
                    pieces.append(ln[start:end])
                    start = end - overlap if overlap > 0 else end
                buf = ""
    if buf:
        pieces.append(buf)

    # Add overlap between chunks (by characters)
    if overlap > 0 and len(pieces) > 1:
        merged = []
        for i, p in enumerate(pieces):
            if i == 0:
                merged.append(p)
            else:
                prev = merged[-1]
                tail = prev[-overlap:] if len(prev) > overlap else prev
                # ensure overlap at the start (not duplicating)
                merged.append(tail + p if not p.startswith(tail) else p)
        pieces = merged

    return pieces

def build_inverted_index(docs: List[str]) -> Tuple[List[Counter], Dict[str, int]]:
    """
    Returns per-doc term counters and document frequencies (df) for all terms.
    """
    per_doc_tf: List[Counter] = []
    df: Dict[str, int] = defaultdict(int)
    for d in docs:
        toks = tokenize(d)
        tf = Counter(toks)
        per_doc_tf.append(tf)
        seen = set(tf.keys())
        for t in seen:
            df[t] += 1
    return per_doc_tf, df

def bm25lite_score(query: str, docs: List[str], per_doc_tf: List[Counter], df: Dict[str, int]) -> List[Tuple[int, float]]:
    """
    Very lightweight scoring: sum_{terms in query} [ idf(term) * (tf / (tf + 1)) ]
    No field length normalization (kept minimal, zero extra deps).
    """
    N = max(1, len(docs))
    q_terms = tokenize(query)
    q_counts = Counter(q_terms)
    unique_terms = list(q_counts.keys())
    scores: List[Tuple[int, float]] = []

    for i, tf in enumerate(per_doc_tf):
        s = 0.0
        for t in unique_terms:
            df_t = df.get(t, 0)
            if df_t == 0:
                continue
            # idf with +1 smoothing
            idf = math.log((N + 1) / (df_t + 1)) + 1.0
            tf_t = tf.get(t, 0)
            if tf_t > 0:
                s += idf * (tf_t / (tf_t + 1.0))
        scores.append((i, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def retrieve_support(sender_id: str, question: str, max_chunks: int = 5) -> Dict[str, Any]:
    """
    Pull the sender's message blob, chunk it, rank chunks by bm25-lite, return top K.
    """
    user = UserMessage.query.filter_by(sender_id=sender_id).first()
    if not user or not (user.messages and user.messages.strip()):
        return {"docs": [], "scores": []}

    docs = split_into_chunks(user.messages, max_chars=800, overlap=80)
    if not docs:
        return {"docs": [], "scores": []}

    per_doc_tf, df = build_inverted_index(docs)
    ranked = bm25lite_score(question, docs, per_doc_tf, df)[:max_chunks]

    top_docs = [docs[i] for (i, _) in ranked]
    top_scores = [float(s) for (_, s) in ranked]
    return {"docs": top_docs, "scores": top_scores}

# =========================
# GenAI wrapper (optional)
# =========================

def generate_with_genai(question: str, support_docs: List[str]) -> str:
    """
    Uses google.genai if GOOGLE_API_KEY is present; otherwise returns a fallback.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # Fallback: simple extractive style answer
        joined = "\n\n".join(support_docs) if support_docs else ""
        if not joined:
            return "Mình chưa có dữ liệu trước đó cho bạn. Hãy nhắn thêm thông tin nhé!"
        return ("(Trả lời trích xuất, không gọi GenAI)\n"
                "Dựa trên lịch sử của bạn, các thông tin liên quan là:\n\n"
                f"{joined}\n\n"
                "Bạn có thể hỏi cụ thể hơn để mình tổng hợp kỹ hơn.")

    client = genai.Client(api_key=api_key)

    # Compose a concise prompt with strict grounding
    system_instructions = (
        "Bạn là một trợ lý ngắn gọn và chính xác. "
        "Chỉ sử dụng thông tin trong phần CONTEXT để trả lời. "
        "Nếu thiếu dữ liệu cần thiết, nói thẳng là không chắc và yêu cầu người dùng cung cấp thêm."
    )

    context_block = "\n\n--- CONTEXT BEGIN ---\n" + \
                    ("\n\n".join(support_docs) if support_docs else "(no prior context)") + \
                    "\n--- CONTEXT END ---\n"

    user_block = f"CÂU HỎI: {question}"

    # The google.genai SDK shape varies across versions; this call path is the current public one.
    # If your environment uses a different method signature, adjust model name or parameters accordingly.
    try:
        resp = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[
                gtypes.Content(role="user", parts=[gtypes.Part.from_text(system_instructions)]),
                gtypes.Content(role="user", parts=[gtypes.Part.from_text(context_block)]),
                gtypes.Content(role="user", parts=[gtypes.Part.from_text(user_block)]),
            ],
            config=gtypes.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=512,
                top_p=0.9,
            ),
        )
        # Extract text safely
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
        # Fallback if parts exist
        if hasattr(resp, "candidates") and resp.candidates:
            parts = getattr(resp.candidates[0], "content", None)
            if parts and getattr(parts, "parts", None):
                texts = [getattr(p, "text", "") for p in parts.parts if getattr(p, "text", "")]
                if texts:
                    return "\n".join(texts).strip()
        return "Không nhận được phản hồi từ GenAI."
    except Exception as e:
        # Never crash the API if GenAI fails
        return f"(GenAI lỗi: {e})"

# =========================
# RAG Route
# =========================

@app.route('/ask', methods=['POST'])
def ask():
    """
    Body: { "sender_id": "...", "question": "..." }
    Returns: { "answer": "...", "support": [...], "scores": [...] }
    """
    data = request.get_json(silent=True) or {}
    sender_id = (data.get("sender_id") or "").strip()
    question = (data.get("question") or "").strip()

    if not sender_id or not question:
        return jsonify({"error": "sender_id and question are required"}), 400

    support = retrieve_support(sender_id=sender_id, question=question, max_chunks=5)
    answer = generate_with_genai(question=question, support_docs=support["docs"])

    return jsonify({
        "answer": answer,
        "support": support.get("docs", []),
        "scores": support.get("scores", []),
    }), 200

# --- Optional: health ---
@app.route('/healthz', methods=['GET'])
def healthz():
    return jsonify({"status": "ok"}), 200

# =========================
# Entrypoint
# =========================

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 3000))  # Render assigns this dynamically
    app.run(host="0.0.0.0", port=port)
