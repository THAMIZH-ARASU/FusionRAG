import os
import json
import requests
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
import time
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
CORS(app)

# Configuration
API_BASE_URL = "http://localhost:8000"
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'md'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page"""
    # Initialize theme if not set
    if 'theme' not in session:
        session['theme'] = 'light'
    return render_template('index.html', theme=session.get('theme', 'light'))

@app.route('/toggle-theme')
def toggle_theme():
    """Toggle between light and dark themes"""
    current_theme = session.get('theme', 'light')
    session['theme'] = 'dark' if current_theme == 'light' else 'light'
    return jsonify({'theme': session['theme']})

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Upload to FastAPI backend
        try:
            with open(filepath, 'rb') as f:
                files = {'file': (filename, f, 'application/octet-stream')}
                response = requests.post(f"{API_BASE_URL}/upload", files=files)
                
                if response.status_code == 200:
                    return jsonify({
                        'success': True,
                        'message': f'File {filename} uploaded successfully!',
                        'filename': filename
                    })
                else:
                    return jsonify({
                        'error': f'Error uploading file: {response.json().get("detail", "Unknown error")}'
                    }), 400
        except Exception as e:
            return jsonify({'error': f'Error uploading file: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/index_documents', methods=['POST'])
def index_documents():
    """Index all uploaded documents"""
    try:
        response = requests.post(f"{API_BASE_URL}/index")
        if response.status_code == 200:
            return jsonify({'success': True, 'message': 'Documents indexed successfully!'})
        else:
            return jsonify({
                'error': f'Error indexing documents: {response.json().get("detail", "Unknown error")}'
            }), 400
    except Exception as e:
        return jsonify({'error': f'Error indexing documents: {str(e)}'}), 500

@app.route('/query', methods=['POST'])
def query_documents():
    """Query documents"""
    data = request.get_json()
    query = data.get('query', '')
    retrieval_methods = data.get('retrieval_methods', ['hybrid'])
    llm_provider = data.get('llm_provider', 'openai')
    max_tokens = data.get('max_tokens', 1000)
    temperature = data.get('temperature', 0.7)
    use_adaptive_retrieval = data.get('use_adaptive_retrieval', True)
    
    if not query.strip():
        return jsonify({'error': 'Query cannot be empty'}), 400
    
    try:
        payload = {
            'query': query,
            'retrieval_methods': retrieval_methods,
            'llm_provider': llm_provider,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'use_adaptive_retrieval': use_adaptive_retrieval
        }
        
        response = requests.post(f"{API_BASE_URL}/query", json=payload)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': response.json().get("detail", "Unknown error")}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/compare', methods=['POST'])
def compare_methods():
    """Compare different RAG methods"""
    data = request.get_json()
    query = data.get('query', '')
    retrieval_methods = data.get('retrieval_methods', ['bm25', 'vector_db', 'hybrid'])
    llm_provider = data.get('llm_provider', 'openai')
    
    if not query.strip():
        return jsonify({'error': 'Query cannot be empty'}), 400
    
    try:
        payload = {
            'query': query,
            'retrieval_methods': retrieval_methods,
            'llm_provider': llm_provider
        }
        
        response = requests.post(f"{API_BASE_URL}/compare", json=payload)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': response.json().get("detail", "Unknown error")}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/documents')
def list_documents():
    """List all documents"""
    try:
        response = requests.get(f"{API_BASE_URL}/documents")
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': response.json().get("detail", "Unknown error")}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Check API health"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': 'API not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 503

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
