# MRIxAI - Flask Application

This is the Flask version of the MRIxAI application, converted from the original Streamlit implementation. It provides a web-based interface for brain MRI analysis with AI-powered tumor detection.

## Features

- Upload brain MRI scans in NIfTI (.nii.gz) or ZIP format
- Advanced 3D tumor segmentation and visualization
- Interactive 3D brain rendering with Plotly
- Slice-by-slice analysis with tumor masking
- AI-powered medical summary generation
- Responsive web interface with maroon/blood-red theme

## Project Structure

```
.
├── flask_app.py          # Main Flask application
├── templates/            # HTML templates
│   └── index.html        # Main page template
├── static/               # Static assets
│   ├── css/              # CSS stylesheets
│   │   └── style.css     # Custom styles
│   └── js/               # JavaScript files
│       └── main.js       # Frontend logic
├── uploads/              # Uploaded MRI files (created at runtime)
├── requirements_flask.txt # Python dependencies
└── README_FLASK.md       # This file
```

## Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements_flask.txt
   ```

2. Set up your Google API key for AI features:
   ```bash
   export GOOGLE_API_KEY="your_api_key_here"
   ```
   
   On Windows:
   ```cmd
   set GOOGLE_API_KEY=your_api_key_here
   ```

## Running the Application

Start the Flask server:
```bash
python flask_app.py
```

The application will be available at `http://localhost:5000`

## API Endpoints

- `GET /` - Serve the main web interface
- `POST /api/upload` - Handle MRI file uploads
- `POST /api/process` - Process MRI with segmentation
- `POST /api/3d-visualization` - Generate 3D visualization data
- `POST /api/slice` - Get a specific MRI slice
- `POST /api/details` - Get detailed analysis results
- `POST /api/summary` - Generate AI medical summary

## Usage

1. Upload a brain MRI file (.nii.gz or .zip format)
2. Adjust detection parameters as needed:
   - Detection Sensitivity: Higher values detect brighter regions
   - Minimum Tumor Size: Minimum size of tumor regions in voxels
3. View results in the different tabs:
   - 3D View: Interactive 3D brain visualization
   - Slices: Slice-by-slice analysis
   - Details: Detailed tumor statistics
   - Summary: AI-generated medical interpretation

## Technology Stack

- **Flask**: Web framework
- **Bootstrap 5**: Frontend styling
- **Plotly.js**: 3D visualization
- **OpenCV-Python**: Image processing
- **NumPy**: Numerical computing
- **Nibabel**: Neuroimaging data handling
- **scikit-image**: Advanced image processing
- **SciPy**: Scientific computing
- **Google Generative AI**: AI summary generation

## Color Scheme

The application uses a maroon/blood-red color scheme:
- Primary color: `#800000` (Maroon)
- Secondary colors: Various shades of red and maroon

## Development

To modify the application:

1. Frontend changes: Edit files in `templates/` and `static/`
2. Backend changes: Edit `flask_app.py`
3. Add new dependencies to `requirements_flask.txt`

## Production Deployment

For production deployment, consider:

1. Using a production WSGI server like Gunicorn
2. Setting up a reverse proxy with Nginx
3. Configuring proper environment variables
4. Adding SSL/TLS encryption
5. Implementing proper error handling and logging

## License

This project is for educational and research purposes. Please consult with medical professionals for any diagnostic needs.