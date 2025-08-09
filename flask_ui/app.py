import os
import json
import requests
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configuration
API_BASE_URL = "http://localhost:8000"
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'md'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Upload to FastAPI backend
        try:
            with open(filepath, 'rb') as f:
                files = {'file': (filename, f, 'application/octet-stream')}
                response = requests.post(f"{API_BASE_URL}/upload", files=files)
                
                if response.status_code == 200:
                    flash(f'File {filename} uploaded successfully!')
                else:
                    flash(f'Error uploading file: {response.json().get("detail", "Unknown error")}')
        except Exception as e:
            flash(f'Error uploading file: {str(e)}')
        
        return redirect(url_for('index'))
    
    flash('Invalid file type')
    return redirect(request.url)

@app.route('/index_documents', methods=['POST'])
def index_documents():
    """Index all uploaded documents"""
    try:
        response = requests.post(f"{API_BASE_URL}/index")
        if response.status_code == 200:
            flash('Documents indexed successfully!')
        else:
            flash(f'Error indexing documents: {response.json().get("detail", "Unknown error")}')
    except Exception as e:
        flash(f'Error indexing documents: {str(e)}')
    
    return redirect(url_for('index'))

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
