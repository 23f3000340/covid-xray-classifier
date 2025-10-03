from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import json
import io

app = Flask(__name__)

# Load model info
with open('model_info.json', 'r') as f:
    model_info = json.load(f)

# Define model class
class ViTClassifier(torch.nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)

# Load model
print("Loading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ViTClassifier(num_classes=3)
model.load_state_dict(torch.load('vit_covid_model.pth', map_location=device))
model.eval()
model = model.to(device)
print("Model loaded successfully!")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return render_template('index.html', model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Load and preprocess image
        img = Image.open(file.stream).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            pred_idx = probs.argmax(1).item()
            confidence = probs[0, pred_idx].item()
        
        # Get all probabilities
        all_probs = {model_info['classes'][i]: float(probs[0, i].item()) 
                     for i in range(len(model_info['classes']))}
        
        return jsonify({
            'prediction': model_info['classes'][pred_idx],
            'confidence': confidence,
            'all_probabilities': all_probs
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
