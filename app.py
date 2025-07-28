from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

import torch
import cv2
import numpy as np
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import os
import sys
import pathlib

app = Flask(__name__)
CORS(app)

# Model and classes
model = None
CLASS_NAMES = ['W180', 'W300', 'W500']  # Your cashew kernel classes


def load_model():
    """Load the trained YOLOv5 model with Windows compatibility patch"""
    global model
    try:
        # Patch PosixPath to WindowsPath if on Windows
        if os.name == 'nt':
            pathlib.PosixPath = pathlib.WindowsPath
        # Try torch.hub YOLOv5 loading
        model = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path=os.path.abspath('C:/Users/manoj/Documents/Real-Time-Cashew-Kernel-Classification-Using-Deep-Learning/best.pt'),
            force_reload=False,  # Only download once, use cache after
            trust_repo=True
        )
        model.eval()
        print("✅ YOLOv5 model loaded successfully")
        print(f"Model classes: {model.names}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        # Try direct torch.load as fallback
        try:
            checkpoint = torch.load('best.pt', map_location='cpu')
            print("✅ Model loaded with torch.load fallback")
            return checkpoint
        except Exception as e2:
            print(f"❌ Fallback loading also failed: {e2}")
            return None

# Load model at startup
model = load_model()

def preprocess_image(image):
    """Preprocess image for YOLOv5 inference"""
    # Resize image to 640x640 (YOLOv5 standard)
    img_resized = image.resize((640, 640))
    # Convert to numpy array and normalize
    img_array = np.array(img_resized) / 255.0
    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().unsqueeze(0)
    return img_tensor, img_resized

def run_inference(image):
    """Run YOLOv5 inference on the image"""
    try:
        # Store original size
        orig_w, orig_h = image.size
        # Resize for inference
        img_resized = image.resize((640, 640))
        results = model(np.array(img_resized))
        predictions = []
        if hasattr(results, 'xyxy') and len(results.xyxy[0]) > 0:
            for det in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = det.tolist()
                # Map bbox back to original image size
                x1 = x1 * orig_w / 640
                x2 = x2 * orig_w / 640
                y1 = y1 * orig_h / 640
                y2 = y2 * orig_h / 640
                class_id = int(cls)
                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
                predictions.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })
        return predictions, image  # Return original image for annotation
    except Exception as e:
        print(f"Error in inference: {e}")
        return [], image

def draw_annotations(image, predictions):
    """Draw bounding boxes and labels on the image"""
    try:
        # Convert PIL image to OpenCV format for drawing
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        for pred in predictions:
            bbox = pred['bbox']
            class_name = pred['class']
            confidence = pred['confidence']
            
            # Extract coordinates
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # Choose color based on class
            if class_name == 'W180':
                color = (0, 255, 0)  # Green
            elif class_name == 'W300':
                color = (255, 0, 0)  # Blue
            elif class_name == 'W500':
                color = (0, 0, 255)  # Red
            else:
                color = (255, 255, 0)  # Yellow
            
            # Draw bounding box
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label = f"{class_name}: {confidence:.2f}"
            
            # Get text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw label background
            cv2.rectangle(img_cv, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(img_cv, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)
        
        # Convert back to PIL
        img_annotated = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        return img_annotated
        
    except Exception as e:
        print(f"Error drawing annotations: {e}")
        return image

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def process_image(image_data):
    """Process image data and return predictions with annotated image"""
    try:
        # Convert base64 to image
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Run actual model inference
        predictions, processed_image = run_inference(image)
        
        # Draw annotations on the processed image
        annotated_image = draw_annotations(processed_image, predictions)
        
        # Convert annotated image to base64
        annotated_image_b64 = image_to_base64(annotated_image)
        
        # Calculate model accuracy based on average confidence and detection quality
        model_accuracy = 0.0
        if predictions:
            # Calculate average confidence
            avg_confidence = sum(pred['confidence'] for pred in predictions) / len(predictions)
            
            # Consider high confidence detections (>0.7) as more reliable
            high_confidence_detections = [pred for pred in predictions if pred['confidence'] > 0.7]
            confidence_score = len(high_confidence_detections) / len(predictions) if predictions else 0
            
            # Combine average confidence with high-confidence ratio
            model_accuracy = (avg_confidence + confidence_score) / 2
        
        return {
            'success': True,
            'predictions': predictions,
            'total_detections': len(predictions),
            'model_accuracy': model_accuracy,
            'annotated_image': annotated_image_b64,
            'message': f'Detected {len(predictions)} cashew kernels with annotations.'
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
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model is not None,
        'classes': CLASS_NAMES,
        'message': 'Cashew Kernel Classification Service is running'
    })

@app.route('/model-info')
def model_info():
    """Get model information"""
    if model is not None:
        return jsonify({
            'model_type': str(type(model)),
            'classes': CLASS_NAMES,
            'message': 'YOLOv5 model loaded successfully'
        })
    else:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Please check your best.pt file'
        })

if __name__ == '__main__':
    print("🌰 Cashew Kernel Classification Application")
    print("=" * 50)
    
    if model is not None:
        print("✅ YOLOv5 model loaded successfully!")
        print(f"📊 Classes: {CLASS_NAMES}")
        print("📱 Open your browser and go to: http://localhost:5000")
        print("🔍 Check model info at: http://localhost:5000/model-info")
    else:
        print("❌ Model loading failed!")
        print("Please ensure 'best.pt' file is in the current directory")
    
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)