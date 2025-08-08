#!/usr/bin/env python3
"""
Flask web application for Image to SVG converter
"""

from flask import Flask, render_template, request, send_file, jsonify, Response
from werkzeug.utils import secure_filename
import os
import tempfile
from pathlib import Path
import base64
from io import BytesIO, StringIO
from PIL import Image
import sys
import json
import time
import atexit
import shutil

# Import our converter
from img2svg import ImageToSVGConverter, SVGColorProcessor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_EXTENSIONS'] = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg'}

# Create temp directory in project
TEMP_DIR = Path(__file__).parent / 'temp'
TEMP_DIR.mkdir(exist_ok=True)

# Cleanup function for temp files
def cleanup_temp_files():
    """Clean up old temp files on startup and shutdown"""
    if TEMP_DIR.exists():
        for file in TEMP_DIR.glob('tmp*'):
            try:
                if file.is_file():
                    file.unlink()
            except:
                pass

# Clean up on startup
cleanup_temp_files()

# Register cleanup on shutdown
atexit.register(cleanup_temp_files)

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

class ProgressCapture:
    """Capture stdout to send as progress updates"""
    def __init__(self):
        self.logs = []
        self.original_stdout = sys.stdout
        
    def write(self, text):
        if text.strip():
            self.logs.append(text.strip())
        self.original_stdout.write(text)
        
    def flush(self):
        self.original_stdout.flush()
        
    def get_logs(self):
        return self.logs

@app.route('/convert', methods=['POST'])
def convert():
    """Handle image upload and conversion"""
    
    # Extract all request data before entering the generator
    # Check if file was uploaded
    if 'file' not in request.files:
        return Response(
            f"data: {json.dumps({'type': 'error', 'message': 'No file uploaded'})}\n\n",
            mimetype="text/event-stream"
        )
    
    file = request.files['file']
    if file.filename == '':
        return Response(
            f"data: {json.dumps({'type': 'error', 'message': 'No file selected'})}\n\n",
            mimetype="text/event-stream"
        )
    
    # Get settings from form
    n_colors = int(request.form.get('colors', 8))
    method = request.form.get('method', 'kmeans')
    threshold = int(request.form.get('threshold', 128))
    simplify = request.form.get('simplify', 'true') == 'true'
    include_background = request.form.get('includeBackground', 'false') == 'true'
    
    # Get suggested colors if provided
    suggested_colors_json = request.form.get('suggestedColors', '[]')
    try:
        suggested_colors = json.loads(suggested_colors_json)
        # Convert hex colors to RGB tuples
        suggested_colors_rgb = []
        for color in suggested_colors:
            if color.startswith('#'):
                color = color[1:]
            r = int(color[0:2], 16)
            g = int(color[2:4], 16)
            b = int(color[4:6], 16)
            suggested_colors_rgb.append((r, g, b))
    except:
        suggested_colors_rgb = []
    
    # Save uploaded file temporarily before streaming
    filename = secure_filename(file.filename)
    file_ext = Path(filename).suffix.lower()
    
    if file_ext not in app.config['UPLOAD_EXTENSIONS']:
        return Response(
            f"data: {json.dumps({'type': 'error', 'message': 'Invalid file type'})}\n\n",
            mimetype="text/event-stream"
        )
    
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False, dir=TEMP_DIR) as tmp_input:
        file.save(tmp_input.name)
        input_path = tmp_input.name
    
    def generate():
        try:
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting conversion...'})}\n\n"
            time.sleep(0.1)
            
            yield f"data: {json.dumps({'type': 'status', 'message': f'Settings: {n_colors} colors, {method} method'})}\n\n"
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Processing file...'})}\n\n"
            
            # Create output file
            with tempfile.NamedTemporaryFile(suffix='.svg', delete=False, dir=TEMP_DIR) as tmp_output:
                output_path = tmp_output.name
            
            # Capture progress output
            progress_capture = ProgressCapture()
            sys.stdout = progress_capture
            
            try:
                if file_ext == '.svg':
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Processing SVG file...'})}\n\n"
                    # Process existing SVG
                    processor = SVGColorProcessor()
                    processor.quantize_colors(input_path, output_path, n_colors)
                else:
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Converting image to SVG...'})}\n\n"
                    if suggested_colors_rgb:
                        yield f"data: {json.dumps({'type': 'status', 'message': f'Using {len(suggested_colors_rgb)} suggested colors'})}\n\n"
                    # Convert image to SVG
                    converter = ImageToSVGConverter(
                        n_colors=n_colors,
                        method=method,
                        simplify=simplify,
                        threshold=threshold,
                        include_background=include_background,
                        suggested_colors=suggested_colors_rgb
                    )
                    converter.convert(input_path, output_path)
                
                # Send captured logs
                for log in progress_capture.get_logs():
                    yield f"data: {json.dumps({'type': 'log', 'message': log})}\n\n"
                    time.sleep(0.05)  # Small delay to make progress visible
                
                # Restore stdout
                sys.stdout = progress_capture.original_stdout
                
                yield f"data: {json.dumps({'type': 'status', 'message': 'Reading output file...'})}\n\n"
                
                # Read the output SVG
                with open(output_path, 'r') as f:
                    svg_content = f.read()
                
                yield f"data: {json.dumps({'type': 'complete', 'svg': svg_content})}\n\n"
                
            finally:
                # Restore stdout
                sys.stdout = progress_capture.original_stdout
                # Clean up temp files
                try:
                    if os.path.exists(input_path):
                        os.unlink(input_path)
                    if os.path.exists(output_path):
                        os.unlink(output_path)
                except Exception as e:
                    print(f"Warning: Could not clean up temp files: {e}")
            
        except Exception as e:
            sys.stdout = sys.__stdout__  # Ensure stdout is restored
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return Response(generate(), mimetype="text/event-stream")

@app.route('/download-svg', methods=['POST'])
def download_svg():
    """Generate and download SVG file"""
    tmp_path = None
    try:
        svg_content = request.json.get('svg', '')
        if not svg_content:
            return jsonify({'error': 'No SVG content provided'}), 400
        
        # Create temporary file for download
        with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False, dir=TEMP_DIR) as tmp:
            tmp.write(svg_content)
            tmp_path = tmp.name
        
        # Send file and schedule cleanup
        def cleanup():
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except:
                pass
        
        response = send_file(
            tmp_path,
            mimetype='image/svg+xml',
            as_attachment=True,
            download_name='converted.svg'
        )
        response.call_on_close(cleanup)
        return response
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Run the app
    app.run(debug=True, port=5000)