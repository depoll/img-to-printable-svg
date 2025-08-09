#!/usr/bin/env python3
import os
import io
import subprocess
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
import traceback

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['OUTPUT_FOLDER'] = '/tmp/outputs'

# Ensure upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'svg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def convert():
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, GIF, BMP, SVG'}), 400
        
        # Get parameters
        num_colors = request.form.get('colors', '8')
        method = request.form.get('method', 'kmeans')
        threshold = request.form.get('threshold', '128')
        simplify = request.form.get('simplify', 'true') != 'false'
        remove_gradients = request.form.get('remove_gradients', 'false') == 'true'
        quantize_only = request.form.get('quantize_only', 'false') == 'true'
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Generate output filename
        output_filename = f"{os.path.splitext(filename)[0]}_converted.svg"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Build command
        cmd = ['python', 'img2svg.py', input_path, output_path]
        cmd.extend(['-c', num_colors])
        cmd.extend(['-m', method])
        cmd.extend(['-t', threshold])
        
        if not simplify:
            cmd.append('--no-simplify')
        if remove_gradients:
            cmd.append('--remove-gradients')
        if quantize_only:
            cmd.append('--quantize-only')
        
        # Run conversion
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or 'Unknown error'
            return jsonify({'error': f'Conversion failed: {error_msg}'}), 500
        
        # Read the output SVG
        with open(output_path, 'r') as f:
            svg_content = f.read()
        
        # Clean up temporary files
        os.remove(input_path)
        os.remove(output_path)
        
        # Return SVG content
        return jsonify({
            'success': True,
            'svg': svg_content,
            'filename': output_filename
        })
        
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Conversion timeout - file may be too large'}), 500
    except Exception as e:
        app.logger.error(f"Conversion error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/download', methods=['POST'])
def download():
    try:
        svg_content = request.json.get('svg', '')
        filename = request.json.get('filename', 'converted.svg')
        
        if not svg_content:
            return jsonify({'error': 'No SVG content provided'}), 400
        
        # Create a BytesIO object
        svg_bytes = io.BytesIO(svg_content.encode('utf-8'))
        svg_bytes.seek(0)
        
        return send_file(
            svg_bytes,
            mimetype='image/svg+xml',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)