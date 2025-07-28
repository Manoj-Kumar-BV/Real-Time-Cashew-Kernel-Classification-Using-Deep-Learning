from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import cv2
import numpy as np
import base64
import io
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
model.eval()

def process_image(image_data):
    """Process image data and return predictions"""
    try:
        # Convert base64 to image
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Run inference
        results = model(img_array)
        
        # Extract predictions
        predictions = []
        if len(results.xyxy[0]) > 0:
            for det in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = det.tolist()
                class_name = model.names[int(cls)]
                predictions.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })
        
        return {
            'success': True,
            'predictions': predictions,
            'total_detections': len(predictions)
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    """Endpoint for image classification"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'error': 'No image data provided'})
        
        result = process_image(image_data)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 