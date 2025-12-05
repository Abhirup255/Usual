"""
Project: Alzheimer's Disease Detection Using Deep Learning
Semester: 7th (Project-II)
Subject Code: PROJ-CS781
Department: Computer Science & Engineering, Academy of Technology

Team Members:
- Sayan Mandal (16900122012)
- Abhirup Nandi (16900122039)
- Bikramjit Ghosh (16900123194)
- Anik Nandi (16900123192)
- Aritra Chowdhury (16900122006)

Module: System Design - Web Interface (Report Chapter 5.1, Step 4)
Description: This Flask server handles HTTP POST requests, saves uploaded files securely,
and feeds them to the InceptionV3 model for inference.
"""

from flask import Flask, render_template, request
import os
import label_image  # Importing the inference engine

app = Flask(__name__)

# CONFIGURATION
# As per Report Section 5.4 (Database Design):
# "User uploaded images are stored temporarily... and cleaned up immediately." [cite: 203]
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    """
    Serves the UI where users can upload MRI scans.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_file():
    """
    Implements the Prediction Flow Algorithm (Report Section 5.3)[cite: 182].
    Step 3: Receive User Image Input via Web Form.
    Step 4-9: Handled by label_image.py routine.
    """
    if request.method == 'POST':
        # 1. Validation
        file = request.files['file']
        if not file:
            return "No file uploaded", 400

        # 2. Save File Temporarily (Database Design - File-based storage) [cite: 202]
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # 3. Inference (Connecting to InceptionV3) [cite: 197]
        # We pass the image to the 'label_image' module which contains the graph logic
        result = label_image.classify_image(file_path)
        
        # 4. Display Result [cite: 191]
        # In a real deployment, we might clean up the file here: os.remove(file_path)
        return result

if __name__ == "__main__":
    # Running on standard hardware as per Feasibility Study [cite: 152]
    app.run(debug=True, port=4555)