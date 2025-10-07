from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Configure SQLite (or change URI for MySQL/PostgreSQL)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///messages.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Model ---
class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.String(50))
    text = db.Column(db.String(255))

    def to_dict(self):
        return {"sender_id": self.sender_id, "text": self.text}

# Create table
with app.app_context():
    db.create_all()

# --- API Routes ---
@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()

    try:
        # Extract sender id and text safely
        entry = data[0]["body"]["entry"][0]
        message_data = entry["messaging"][0]
        sender_id = message_data["sender"]["id"]
        text = message_data["message"]["text"]

        # Save to database
        msg = Message(sender_id=sender_id, text=text)
        db.session.add(msg)
        db.session.commit()

        return jsonify({"message": "Saved successfully"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/messages', methods=['GET'])
def get_messages():
    messages = Message.query.all()
    return jsonify([m.to_dict() for m in messages])

if __name__ == '__main__':
    app.run(debug=True)
