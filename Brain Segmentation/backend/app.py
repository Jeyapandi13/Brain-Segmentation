from flask import Flask, render_template, request, url_for, redirect, flash
import os
import uuid
from werkzeug.utils import secure_filename
from model.predict import predict_tumor

# Initialize Flask app
app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

app.secret_key = 'your-secret-key-here'  # Required for flash messages

# Configuration
UPLOAD_FOLDER = os.path.join(app.static_folder, 'uploads')
SEGMENTED_FOLDER = os.path.join(app.static_folder, 'segmented')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'dcm'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEGMENTED_FOLDER'] = SEGMENTED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create necessary folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SEGMENTED_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_unique_filename(filename):
    """Generate a unique filename to prevent overwrites"""
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'jpg'
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    return unique_name


@app.route("/")
def index():
    """Home page with upload form"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle MRI image upload and prediction"""
    
    # Check if file was uploaded
    if 'mri' not in request.files:
        flash('No file was uploaded!', 'error')
        return redirect(url_for('index'))
    
    file = request.files['mri']
    
    # Check if file was selected
    if file.filename == '':
        flash('No file was selected!', 'error')
        return redirect(url_for('index'))
    
    # Validate file type
    if not allowed_file(file.filename):
        flash('Invalid file type! Please upload PNG, JPG, JPEG, or GIF.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Generate unique filename and save
        original_filename = secure_filename(file.filename)
        unique_filename = generate_unique_filename(original_filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Run AI prediction
        tumor, tumor_type, risk, segmented_filename = predict_tumor(
            filepath,
            app.config['SEGMENTED_FOLDER'],
            unique_filename
        )
        
        # Generate URLs for templates
        uploaded_image_url = url_for('static', filename=f'uploads/{unique_filename}')
        segmented_image_url = url_for('static', filename=f'segmented/{segmented_filename}')
        
        return render_template(
            "result.html",
            tumor=tumor,
            tumor_type=tumor_type,
            risk=risk,
            uploaded_image=uploaded_image_url,
            segmented_image=segmented_image_url,
            filename=original_filename
        )
        
    except Exception as e:
        flash(f'An error occurred during analysis: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route("/about")
def about():
    """About page"""
    return render_template("about.html")


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File is too large! Maximum size is 16MB.', 'error')
    return redirect(url_for('index'))


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    flash('An internal error occurred. Please try again.', 'error')
    return redirect(url_for('index'))


if __name__ == "__main__":
    print("=" * 50)
    print("🧠 Brain Tumor Detection AI")
    print("=" * 50)
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"📁 Segmented folder: {SEGMENTED_FOLDER}")
    print("🌐 Starting server at http://localhost:5000")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)