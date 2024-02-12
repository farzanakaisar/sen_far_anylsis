from flask import Flask, request, jsonify
from transformers import DistilBertModel, DistilBertTokenizer

app = Flask(__name__)

# Load pre-trained DistilBERT model and tokenizer
distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Define an endpoint for text classification
@app.route('/classify', methods=['POST'])
def classify_text():
    try:
        # Get input text from the request
        text = request.json['text']
        
        # Tokenize input text
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

        # Get DistilBERT embeddings
        with torch.no_grad():
            outputs = distilbert_model(**inputs)
            embeddings = outputs.last_hidden_state

        # Perform classification or other tasks if needed
        # Example: classification using embeddings

        # Return the response
        return jsonify({'embeddings': embeddings.tolist()}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
