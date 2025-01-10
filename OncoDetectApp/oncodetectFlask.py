from flask import Flask, request, render_template, url_for
from torchvision import transforms
import torch 
from OncoDetect import SimpleCNN
from PIL import Image

app = Flask(__name__)

model = SimpleCNN()
model.load_state_dict(torch.load('model.pth', map_location = torch.device('cpu')))
model.eval()

#Define same transformations present in CNN model 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html') #connect front end file 

@app.route('/predict', methods = ['POST'])
def predict(): 
    #Check if file uploaded
    if 'file' not in request.files: 
        return 'No file'
    file = request.files['file']
    #if no file selected
    if file.filename == '':
        return 'No selected file'
    
    #open, convert uploaded image to RGB, output prediction
    class_names = {'0': 'Normal', '1': 'Malignant'}
    if file: 
        image = Image.open(file.stream).convert('RGB') #assign variable image to the uploaded file 
        image = transform(image).unsqueeze(0) #apply transformations to uploaded image 
        with torch.no_grad(): 
            prediction = model(image).argmax().item()
        predicted_class = class_names[str(prediction)]
        return f'Predicted Class: {predicted_class}' #output prediction 
    
if __name__ == '__main__':
    app.run(debug = True)