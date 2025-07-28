# 🌰 Real-Time Cashew Kernel Classification

A web-based application for real-time classification of cashew kernels using YOLOv5 deep learning model. The application provides both webcam access and image upload functionality for classifying cashew kernels.

## Features

- **📷 Webcam Integration**: Real-time camera access for instant classification
- **📁 Image Upload**: Drag & drop or browse files for classification
- **🤖 YOLOv5 Model**: Advanced deep learning model for accurate classification
- **🎨 Modern UI**: Beautiful, responsive design with smooth animations
- **📊 Results Display**: Clear visualization of classification results with confidence scores
- **📱 Mobile Responsive**: Works seamlessly on desktop and mobile devices

## Prerequisites

- Python 3.8 or higher
- Webcam (for webcam functionality)
- Modern web browser with camera access permissions

## Installation

1. **Clone or download the project files**
   ```bash
   # Make sure you have the following files in your directory:
   # - app.py
   # - best.pt (your trained YOLOv5 model)
   # - requirements.txt
   # - templates/index.html
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify your model file**
   - Ensure `best.pt` is in the same directory as `app.py`
   - This should be your trained YOLOv5 model for cashew kernel classification

## Usage

1. **Start the application**
   ```bash
   python app.py
   ```

2. **Access the web interface**
   - Open your web browser
   - Navigate to `http://localhost:5000`
   - The application will load with a modern interface

3. **Using the application**

   ### Webcam Classification
   - Click the "📷 Webcam" tab
   - Click "Start Camera" to enable webcam access
   - Position cashew kernels in view
   - Click "📸 Capture & Classify" to analyze the image
   - View results with confidence scores

   ### Image Upload Classification
   - Click the "📁 Upload Image" tab
   - Drag and drop an image or click to browse files
   - Supported formats: JPG, PNG, JPEG
   - The image will be automatically classified
   - View detailed results with bounding boxes and confidence scores

## API Endpoints

- `GET /` - Main web interface
- `POST /classify` - Image classification endpoint
- `GET /health` - Health check endpoint

## Model Information

The application uses a custom-trained YOLOv5 model (`best.pt`) specifically designed for cashew kernel classification. The model can detect and classify different types of cashew kernels with high accuracy.

## Technical Details

### Backend (Flask)
- **Framework**: Flask with CORS support
- **Model**: YOLOv5 custom model
- **Image Processing**: OpenCV and PIL
- **Response Format**: JSON with predictions and confidence scores

### Frontend (HTML/CSS/JavaScript)
- **Design**: Modern, responsive UI with gradient backgrounds
- **Features**: Tab-based interface, drag & drop, webcam integration
- **Animations**: Smooth transitions and loading indicators
- **Compatibility**: Works on all modern browsers

## File Structure

```
Real-Time-Cashew-Kernel-Classification-Using-Deep-Learning/
├── app.py                 # Flask backend application
├── best.pt               # Trained YOLOv5 model
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html       # Frontend web interface
└── README.md            # This file
```

## Troubleshooting

### Common Issues

1. **Camera not working**
   - Ensure your browser allows camera access
   - Check if another application is using the camera
   - Try refreshing the page

2. **Model loading errors**
   - Verify `best.pt` file exists in the project directory
   - Check if all dependencies are installed correctly
   - Ensure sufficient RAM for model loading

3. **Installation issues**
   - Use a virtual environment: `python -m venv venv`
   - Activate it: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/Mac)
   - Install dependencies: `pip install -r requirements.txt`

4. **Port already in use**
   - Change the port in `app.py`: `app.run(debug=True, host='0.0.0.0', port=5001)`
   - Or kill the process using port 5000

### Performance Tips

- Use a GPU for faster inference (if available)
- Close unnecessary browser tabs
- Ensure good lighting for webcam captures
- Use high-quality images for better results

## Development

### Adding New Features

1. **Modify the backend** (`app.py`):
   - Add new routes for additional functionality
   - Enhance the model processing logic
   - Add new API endpoints

2. **Update the frontend** (`templates/index.html`):
   - Modify the UI design and layout
   - Add new JavaScript functionality
   - Enhance user experience

3. **Model improvements**:
   - Retrain the YOLOv5 model with more data
   - Fine-tune hyperparameters
   - Add new cashew kernel classes

## License

This project is for educational and research purposes. Please ensure you have the necessary permissions for any commercial use.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the application.

## Support

For technical support or questions, please check the troubleshooting section above or create an issue in the project repository.

---

**Note**: This application requires a trained YOLOv5 model (`best.pt`) for cashew kernel classification. Make sure you have the appropriate model file before running the application. 