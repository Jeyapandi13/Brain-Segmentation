🧠 Brain Tumor Detection Using Deep Learning

A web-based Brain Tumor Detection System built using Flask, PyTorch, and ResNet18 that predicts the type of brain tumor from MRI images.

The system allows users to upload MRI images and receive:

Tumor Detection (Yes / No)

Tumor Type Classification

Risk Probability Percentage

🚀 Features

MRI Image Upload via Web Interface

Deep Learning Tumor Classification

4 Tumor Classes

Risk Probability Output

Flask Web Application

PyTorch Deep Learning Model

Pretrained ResNet18 Architecture

🧬 Tumor Types Supported

The model classifies MRI scans into:

glioma
meningioma
pituitary
no_tumor
🏗 Project Architecture
User Upload MRI
       ↓
Flask Backend
       ↓
Image Preprocessing
       ↓
ResNet18 Deep Learning Model
       ↓
Tumor Classification
       ↓
Risk Percentage Output
📂 Project Folder Structure
Brain Segmentation
│
├── backend
│   └── app.py
│
├── model
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   └── tumor_model.pth
│
├── dataset
│   ├── glioma
│   ├── meningioma
│   ├── pituitary
│   └── no_tumor
│
├── frontend
│   ├── css
│   │   └── style.css
│   │
│   ├── js
│   │   └── script.js
│   │
│   └── templates
│       ├── index.html
│       └── result.html
│
├── uploads
│
├── run.py
├── requirements.txt
└── README.md
🛠 Technologies Used

Python

Flask

PyTorch

TorchVision

OpenCV

HTML

CSS

JavaScript

⚙️ Installation Guide

Clone the repository:

git clone https://github.com/yourusername/brain-tumor-detection.git

Move into project directory:

cd brain-tumor-detection
📦 Create Virtual Environment
python -m venv venv

Activate environment:

Windows
venv\Scripts\activate
Linux / Mac
source venv/bin/activate

python -m pip install --upgrade pip

📥 Install Dependencies
pip install flask
pip install torch torchvision
pip install pillow
pip install numpy
pip install opencv-python

Or install everything:

python -m pip install -r requirements.txt

📊 Dataset Preparation

Create dataset folder:

dataset/

Structure:

dataset

 added the all images of brain tumour diseases 

Each folder must contain MRI images for that tumor class.

🧠 Train the Deep Learning Model

Run training script:

python model/train.py

Training output example:

Training Started...
Epoch 1 completed
Epoch 2 completed
Epoch 3 completed
Epoch 4 completed
Epoch 5 completed
Model Saved!

After training, the model will be saved as:

model/tumor_model.pth
🌐 Run the Web Application

Start the Flask server:

python run.py

Server will start at:

http://127.0.0.1:5000

Open this URL in your browser.

🖥 Using the System

Open the website

Upload MRI scan image

Click Predict

System returns:

Example:

Tumor Detected: Yes
Tumor Type: Glioma
Risk Percentage: 92.5%
📈 Model Details

Model Architecture:

ResNet18

Training Parameters:

Image Size : 224x224
Optimizer  : Adam
Loss       : CrossEntropyLoss
Epochs     : 5-20
Classes    : 4
📊 Prediction Pipeline
MRI Image
   ↓
Resize (224x224)
   ↓
Tensor Conversion
   ↓
ResNet18 Model
   ↓
Softmax Probability
   ↓
Tumor Type + Risk %
⚠️ Limitations

Model accuracy depends on dataset size

Small datasets may reduce performance

Not intended for medical diagnosis

🔮 Future Improvements

Brain Tumor Segmentation using U-Net

Tumor Region Highlighting

MRI Heatmap Visualization

Doctor Dashboard Interface

Patient Report Generation

Deploy with Docker / Cloud

📸 Example Output
Tumor Detected : Yes
Tumor Type : Pituitary
Risk : 93%
👨‍💻 Author

Developed by Jeyapandi

Deep Learning Project – Brain Tumor Detection using PyTorch & Flask.
