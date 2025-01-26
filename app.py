from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import google.generativeai as genai
from flask_cors import CORS  # Import Flask-CORS

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all origins (or specify specific domains)
CORS(app)  # This allows all origins by default

# Get the API key from environment variable
api_key = os.getenv("API_KEY")
if api_key is None:
    raise ValueError("API key not found. Please check your .env file.")

# Configure the Gemini model with the loaded API key
genai.configure(api_key=api_key)

# Create the model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Instantiate the Generative Model
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)


# Define a route for AI response
@app.route('/chat', methods=['POST'])
def chat():
    # Try to get the JSON payload from the request
    try:
        data = request.json
        if not data or "message" not in data:
            return jsonify({"error": "Invalid input, 'message' field is required."}), 400

        message = data["message"]

        # Start a chat session and send the message to the model
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(message)

        # Return the response from the AI
        return jsonify({"response": response.text})

    except Exception as e:
        # Return an error response if something goes wrong
        return jsonify({"error": f"Failed to process request: {str(e)}"}), 500

@app.route('/')
def home():
    return "Flask server is running! Use the /chat route to interact with the AI."

if __name__ == '__main__':
    # Start the Flask app
    app.run(host='0.0.0.0', port=8080)
    # app.run(debug=True)
