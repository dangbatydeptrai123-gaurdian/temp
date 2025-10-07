from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

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

# --- Helper: extract id + text ---
def extract_message(data):
    entry = data[0]["body"]["entry"][0]
    msg = entry["messaging"][0]
    return msg["sender"]["id"], msg["message"]["text"]

# --- Routes ---
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


if __name__ == '__main__':
    app.run(debug=True)
