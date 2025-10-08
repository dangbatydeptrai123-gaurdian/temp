from flask import Flask, request, jsonify, abort
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
from datetime import datetime
import os
import logging
import json

app = Flask(__name__)

# --- Config ---
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///messages.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2 MB guard
READ_API_TOKEN = os.getenv('READ_API_TOKEN')  # simple bearer token for /messages
VERIFY_TOKEN = os.getenv('VERIFY_TOKEN')      # for webhook verification (FB style)

db = SQLAlchemy(app)

# --- Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.String(64), unique=True, index=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), index=True, nullable=False)
    mid = db.Column(db.String(128), unique=True, index=True)  # platform message id (for idempotency)
    text = db.Column(db.Text)  # message text (if any)
    raw = db.Column(db.Text)   # raw JSON payload (optional for debugging/audit)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    user = db.relationship('User', backref=db.backref('messages', lazy='dynamic', cascade='all, delete-orphan'))

with app.app_context():
    db.create_all()

# --- Helpers ---
def get_or_create_user(sender_id: str) -> User:
    user = User.query.filter_by(sender_id=sender_id).first()
    if user:
        return user
    user = User(sender_id=sender_id)
    db.session.add(user)
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        # in case of race, retrieve the one created by the other request
        user = User.query.filter_by(sender_id=sender_id).first()
    return user

def parse_messenger_events(payload: dict):
    """
    Yield (sender_id, mid, text, raw_json_str) for each *text* message.
    Ignores non-text (attachments), echoes, and other event types.
    """
    if not isinstance(payload, dict):
        return

    entries = payload.get('entry', [])
    for entry in entries:
        for evt in entry.get('messaging', []):
            sender = (evt.get('sender') or {}).get('id')
            message = evt.get('message')
            # Skip non-message events (postbacks, deliveries, etc.)
            if not sender or not message:
                continue
            # Skip echoes
            if message.get('is_echo'):
                continue
            text = message.get('text')
            # Only handle text here; you could extend for attachments
            if text is None:
                continue
            mid = message.get('mid')  # idempotency key
            yield sender, mid, text, json.dumps(evt, ensure_ascii=False)

def bearer_auth_ok(req: request) -> bool:
    if not READ_API_TOKEN:
        return False
    auth = req.headers.get('Authorization', '')
    return auth == f'Bearer {READ_API_TOKEN}'

# --- Routes ---

# Optional: Webhook verification (Facebook-style GET)
@app.route('/webhook', methods=['GET'])
def verify_webhook():
    # FB sends: hub.mode, hub.verify_token, hub.challenge
    mode = request.args.get('hub.mode')
    token = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')
    if mode == 'subscribe' and token and token == VERIFY_TOKEN:
        return challenge or '', 200
    return 'Forbidden', 403

# Ingest webhook (POST)
@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        payload = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"error": "Invalid or missing JSON"}), 400

    if not payload:
        return jsonify({"error": "Empty payload"}), 400

    saved = 0
    try:
        for sender_id, mid, text, raw in parse_messenger_events(payload):
            user = get_or_create_user(sender_id)

            # Idempotency: skip if we've already seen this mid
            if mid and Message.query.filter_by(mid=mid).first():
                continue

            msg = Message(user_id=user.id, mid=mid, text=text, raw=raw)
            db.session.add(msg)
            saved += 1

        if saved > 0:
            db.session.commit()
            return jsonify({"message": f"Saved {saved} message(s)"}), 201
        else:
            # Nothing to save (no text messages, duplicates, or non-message events)
            return jsonify({"message": "No new text messages"}), 200

    except IntegrityError as e:
        db.session.rollback()
        return jsonify({"error": "Database integrity error", "detail": str(e)}), 409
    except Exception as e:
        db.session.rollback()
        app.logger.exception("Unhandled error in /webhook")
        return jsonify({"error": "Unhandled error", "detail": str(e)}), 500

# Read messages with simple auth + pagination & filtering
@app.route('/messages', methods=['GET'])
def get_messages():
    if not bearer_auth_ok(request):
        return jsonify({"error": "Unauthorized"}), 401

    sender = request.args.get('sender_id')
    page = max(int(request.args.get('page', 1)), 1)
    per_page = min(max(int(request.args.get('per_page', 20)), 1), 200)

    q = Message.query.join(User)
    if sender:
        q = q.filter(User.sender_id == sender)
    q = q.order_by(Message.created_at.desc())

    items = q.paginate(page=page, per_page=per_page, error_out=False)

    def item_to_dict(m: Message):
        return {
            "id": m.id,
            "sender_id": m.user.sender_id,
            "mid": m.mid,
            "text": m.text,
            "created_at": m.created_at.isoformat(timespec='seconds'),
        }

    return jsonify({
        "page": page,
        "per_page": per_page,
        "total": items.total,
        "messages": [item_to_dict(m) for m in items.items]
    }), 200

# Health probe
@app.route('/healthz', methods=['GET'])
def healthz():
    return jsonify({"status": "ok"}), 200

# --- Entrypoint ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get("PORT", "3000"))
    app.run(host="0.0.0.0", port=port)
