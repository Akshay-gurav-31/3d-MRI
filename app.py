import os
import sys
import json
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, make_response
from flask_cors import CORS
import zipfile
import nibabel as nib
import tempfile
from skimage import measure
import plotly.graph_objects as go
from scipy import ndimage
import google.generativeai as genai
import traceback
from typing import Optional, Tuple, Any, Union, Dict, List
from concurrent.futures import ThreadPoolExecutor
import warnings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

warnings.filterwarnings('ignore')

# For cv2 imports that cause linter errors
try:
    import cv2
except ImportError:
    cv2 = None

# Type stub for nibabel to avoid linter errors
try:
    import nibabel as nib
    from nibabel.nifti1 import Nifti1Image
    from nibabel.filebasedimages import FileBasedImage
except ImportError:
    nib = None
    Nifti1Image = object
    FileBasedImage = object

# Configure Gemini API
# You'll need to set the GOOGLE_API_KEY environment variable
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
else:
    model = None

# Get the absolute path to the app directory
APP_DIR = os.path.dirname(os.path.abspath(__file__))
HYPOTHESES_PATH = os.path.join(APP_DIR, 'hypotheses_utf8.txt')

# Load hypotheses at startup
try:
    with open(HYPOTHESES_PATH, 'r') as f:
        hypotheses = f.read()
except Exception as e:
    print(f"Error loading hypotheses: {str(e)}")
    hypotheses = ""

# Custom error handling class
class BrainAnalysisError(Exception):
    """Custom exception for brain analysis errors"""
    pass

def process_slice(args):
    """Process a single slice for parallel processing"""
    slice_2d, clahe = args
    try:
        # Normalize to 0-255 range
        slice_norm = ((slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d)) * 255).astype(np.uint8)
        # Apply CLAHE
        enhanced = clahe.apply(slice_norm)
        # Convert back to float64 and normalize
        return enhanced.astype(np.float64) / 255.0
    except Exception:
        return slice_2d

def safe_load_volume(file_path: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Safely load volume data from file with proper type conversion"""
    try:
        if file_path.endswith(".nii.gz"):
            try:
                # Load NIfTI file using nibabel
                img = getattr(nib, 'load', lambda x: None)(file_path)
                # Get data using numpy array conversion
                volume = np.asanyarray(getattr(img, 'dataobj', np.array([]))) if img else np.array([])
                
                # Handle different data types
                if volume.dtype == np.dtype('V'):  # Void type
                    volume = volume.astype(np.float64)
                elif volume.dtype != np.float64:
                    volume = volume.astype(np.float64)
                
                # Handle any invalid values
                volume = np.nan_to_num(volume, nan=0.0, posinf=1.0, neginf=0.0)
                
                # Ensure the data is in a valid range
                if np.any(np.isnan(volume)) or np.any(np.isinf(volume)):
                    volume = np.nan_to_num(volume, nan=0.0, posinf=1.0, neginf=0.0)
                
                return volume, None
                
            except Exception as e:
                return None, f"Error loading NIfTI file: {str(e)}"
                
        elif file_path.endswith(".zip"):
            try:
                # Import cv2 locally to avoid import errors
                import cv2
                
                # Create temporary directory for extraction
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Extract zip file
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Get list of image files
                    image_files = sorted([f for f in os.listdir(temp_dir) 
                                       if f.lower().endswith(('.jpg', '.png'))])
                    
                    if not image_files:
                        return None, "No valid images found in zip file"
                    
                    # Read first image to get dimensions
                    imread_func = getattr(cv2, 'imread', lambda x, y: None) if cv2 else lambda x, y: None
                    first_img = imread_func(os.path.join(temp_dir, image_files[0]), 0)  # 0 for grayscale
                    if first_img is None:
                        return None, "Error reading first image"
                    
                    # Create empty volume array with float64 type
                    volume = np.zeros((first_img.shape[0], first_img.shape[1], 
                                     len(image_files)), dtype=np.float64)
                    
                    # Read and process each image
                    for i, fname in enumerate(image_files):
                        img_path = os.path.join(temp_dir, fname)
                        # Use cv2 with proper flags
                        img = imread_func(img_path, 0)  # 0 for grayscale
                        if img is not None:
                            # Convert to float64 and normalize to 0-1
                            volume[:,:,i] = img.astype(np.float64) / 255.0
                    
                    return volume, None
                    
            except Exception as e:
                return None, f"Error processing zip file: {str(e)}"
        else:
            return None, "Unsupported file format"
            
    except Exception as e:
        return None, f"Error loading volume: {str(e)}"

def safe_normalize_volume(volume: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Safely normalize volume data with proper type handling"""
    try:
        if volume is None or volume.size == 0:
            return None, "Empty volume data"
        
        # Ensure volume is float64
        volume = volume.astype(np.float64)
        
        # Handle invalid values
        volume = np.nan_to_num(volume, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Check if volume is constant
        vmin, vmax = np.min(volume), np.max(volume)
        if vmin == vmax:
            return None, "Volume has no variation (all values are the same)"
        
        # Normalize to 0-1 range
        normalized = (volume - vmin) / (vmax - vmin)
        return normalized.astype(np.float64), None
        
    except Exception as e:
        return None, f"Error normalizing volume: {str(e)}"

def safe_segment_tumor(volume: np.ndarray, threshold: float, min_size: int) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Advanced brain tumor segmentation for 3D MRI"""
    try:
        if volume is None or volume.size == 0:
            return None, "Empty volume data"
        
        # Ensure volume is float64
        volume_float = volume.astype(np.float64)
        
        # Validate parameters
        if not 0 <= threshold <= 1:
            return None, "Threshold must be between 0 and 1"
        if min_size < 0:
            return None, "Minimum size must be positive"
        
        try:
            # 1. Brain-specific preprocessing
            smoothed = ndimage.gaussian_filter(volume_float, sigma=1.0)
            
            # 2. Brain-specific normalization
            percentiles = np.percentile(smoothed, [1, 99])
            vmin, vmax = percentiles[0], percentiles[1]
            enhanced = np.clip((smoothed - vmin) / (vmax - vmin), 0, 1)
            
            # 3. Brain-specific tumor detection
            binary = enhanced > threshold
            
            # 4. Brain-specific morphological operations
            kernel = np.ones((3,3,3))
            binary = ndimage.binary_opening(binary, structure=kernel)
            binary = ndimage.binary_closing(binary, structure=kernel)
            
            # 5. Connected component analysis
            # Handle the return value properly to avoid indexing issues
            labeling_result = ndimage.label(binary)
            if isinstance(labeling_result, tuple):
                labeled_array = labeling_result[0]
                num_features = labeling_result[1]
            else:
                labeled_array = labeling_result
                num_features = 0
            # Convert to array before using ravel
            labeled_array_np = np.array(labeled_array)
            sizes = np.bincount(labeled_array_np.ravel())
            mask_sizes = sizes > min_size
            mask_sizes[0] = False  # Remove background
            
            # 6. Brain-specific filtering
            for i in range(1, len(sizes)):
                if mask_sizes[i]:
                    region = (labeled_array_np == i)
                    mean_intensity = np.mean(enhanced[region])
                    region_volume = sizes[i]
                    region_shape = np.sum(region, axis=(0,1))
                    
                    if (mean_intensity < 0.2 or 
                        mean_intensity > 0.8 or 
                        region_volume > volume_float.size * 0.3 or 
                        np.max(region_shape) > volume_float.shape[2] * 0.8):
                        mask_sizes[i] = False
            
            # 7. Create final mask
            # Convert to integer indices for proper indexing
            mask_sizes_array = np.array(mask_sizes, dtype=bool)
            # Create the predicted mask using proper indexing
            predicted_mask = np.zeros_like(labeled_array_np, dtype=np.float64)
            for idx in range(len(mask_sizes_array)):
                if mask_sizes_array[idx]:
                    predicted_mask[labeled_array_np == idx] = 1.0
            
            # 8. Final cleanup
            predicted_mask = ndimage.binary_dilation(predicted_mask, iterations=1)
            predicted_mask = ndimage.binary_erosion(predicted_mask, iterations=1)
            
            return predicted_mask.astype(np.float64), None
            
        except Exception as e:
            return None, f"Error in tumor detection: {str(e)}"
            
    except Exception as e:
        return None, f"Error in segmentation: {str(e)}"

def safe_visualize_slice(slice_img: np.ndarray, mask_slice: Optional[np.ndarray], 
                        contrast: float, brightness: float) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Safely visualize slice with mask overlay"""
    try:
        slice_img = (slice_img * contrast) + brightness
        slice_img = np.clip(slice_img, 0, 1)
        
        # Convert slice to 8-bit for visualization
        slice_uint8 = (slice_img * 255).astype(np.uint8)
        
        if mask_slice is not None:
            # Check if mask_slice is valid
            if mask_slice.size == 0:
                # Return grayscale image when mask is empty
                return slice_uint8, None
                
            if mask_slice.shape != slice_img.shape:
                # Use resize function directly
                mask_slice_uint8 = (mask_slice * 255).astype(np.uint8)
                resize_func = getattr(cv2, 'resize', lambda x, y, interpolation: x) if cv2 else lambda x, y, interpolation: x
                resized_mask = resize_func(mask_slice_uint8, 
                                        (slice_img.shape[1], slice_img.shape[0]), 
                                        interpolation=0)  # 0 for INTER_NEAREST
                mask_slice = resized_mask.astype(np.float64) / 255.0
            
            # Convert mask to boolean (handle None case)
            if mask_slice is not None:
                mask_bool = mask_slice > 0.5
            else:
                mask_bool = np.zeros_like(slice_img, dtype=bool)
            
            # Ensure slice is 3-channel for color operations
            if len(slice_uint8.shape) == 2:
                # Try to convert to RGB using OpenCV, fallback to numpy if needed
                try:
                    if cv2:
                        colored_slice = cv2.cvtColor(slice_uint8, cv2.COLOR_GRAY2RGB)
                    else:
                        colored_slice = np.stack([slice_uint8]*3, axis=-1)
                except Exception:
                    # Fallback if OpenCV fails
                    colored_slice = np.stack([slice_uint8]*3, axis=-1)
            else:
                colored_slice = slice_uint8.copy()
            
            # Apply mask
            masked_slice = colored_slice.copy()
            # Ensure mask_bool is the right shape for indexing
            if mask_bool.ndim == 2 and colored_slice.ndim == 3:
                mask_3d = np.stack([mask_bool]*3, axis=-1)
                masked_slice[mask_3d] = [255, 0, 0]
            else:
                # Handle case where both are 2D or both are 3D
                masked_slice[mask_bool] = [255, 0, 0]
            
            # Blend images
            alpha = 0.7
            try:
                if cv2:
                    result = cv2.addWeighted(colored_slice, 1-alpha, masked_slice, alpha, 0)
                else:
                    # Simple blending using numpy as fallback
                    result = (colored_slice * (1-alpha) + masked_slice * alpha).astype(np.uint8)
            except Exception:
                # Fallback if blending fails
                result = colored_slice
            
            return result, None
        else:
            # Return grayscale image when no mask
            return slice_uint8, None
    except Exception as e:
        return None, f"Error in slice visualization: {str(e)}"

def safe_marching_cubes(data: np.ndarray, level: float) -> Tuple[Optional[Tuple], Optional[str]]:
    """Safely perform marching cubes with error handling"""
    try:
        if data is None or data.size == 0:
            return None, "Empty data for marching cubes"
        
        data = data.astype(np.float64)
        levels = [level, 0.3, 0.7, 0.1, 0.9]
        for l in levels:
            try:
                result = measure.marching_cubes(data, level=l)
                return result, None
            except ValueError:
                continue
        return None, "Could not generate surface with any level"
    except Exception as e:
        return None, f"Error in marching cubes: {str(e)}"

def safe_generate_ai_summary(tumor_volume: float, tumor_percentage: float, 
                           total_volume: float, confidence_score: float) -> Tuple[Optional[str], Optional[str]]:
    """Safely generate AI summary with error handling"""
    try:
        if not model:
            return "AI model not configured. Please set the GOOGLE_API_KEY environment variable.", None
            
        prompt = f"""
        Based on the following brain MRI analysis results, provide a concise medical summary:
        - Tumor Volume: {tumor_volume:,.0f} voxels
        - Tumor Percentage: {tumor_percentage:.2f}%
        - Total Brain Volume: {total_volume:,.0f} voxels
        - Detection Confidence: {confidence_score:.1f}%

        Please provide:
        1. A brief interpretation of these results
        2. Potential clinical significance
        3. Recommended next steps
        Keep the response professional and medical-focused.
        """
        
        generation_config_dict = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        
        # Convert dict to GenerationConfig
        generation_config = genai.types.GenerationConfig(**generation_config_dict)
        
        response = model.generate_content(prompt, generation_config=generation_config)
        return response.text, None
    except Exception as e:
        return None, f"Error generating AI summary: {str(e)}"

# Flask app initialization
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index() -> str:
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file() -> Response:
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            response = jsonify({'error': 'No file provided'})
            response.status_code = 400
            return response
            
        file = request.files['file']
        if file.filename == '':
            response = jsonify({'error': 'No file selected'})
            response.status_code = 400
            return response
            
        if file:
            filename = file.filename
            upload_folder = str(app.config['UPLOAD_FOLDER'])
            file_path = os.path.join(upload_folder, str(filename))
            file.save(file_path)
            
            # Process the uploaded file
            volume, error = safe_load_volume(file_path)
            if error:
                response = jsonify({'error': f"Error loading MRI: {error}"})
                response.status_code = 400
                return response
            
            if volume is not None:
                volume, error = safe_normalize_volume(volume)
                if error:
                    response = jsonify({'error': f"Error processing MRI: {error}"})
                    response.status_code = 400
                    return response
            else:
                response = jsonify({'error': "Failed to load volume data"})
                response.status_code = 400
                return response
            
            # Return success response
            response = jsonify({
                'message': 'File uploaded successfully',
                'file_path': file_path
            })
            response.status_code = 200
            return response
        # If file condition is not met (should not happen but for safety)
        response = jsonify({'error': 'No file provided'})
        response.status_code = 400
        return response
            
    except Exception as e:
        response = jsonify({'error': f"Error uploading file: {str(e)}"})
        response.status_code = 500
        return response
    # Default response in case of unexpected flow
    response = jsonify({'error': 'Unexpected error'})
    response.status_code = 500
    return response

@app.route('/api/process', methods=['POST'])
def process_mri() -> Response:
    """Process MRI with given parameters"""
    try:
        data = request.get_json()
        if not data:
            response = jsonify({'error': 'No JSON data provided'})
            response.status_code = 400
            return response
            
        file_path = data.get('file_path') if data else None
        threshold = float(data.get('threshold', 0.65)) if data else 0.65
        min_size = int(data.get('min_size', 100)) if data else 100
        
        if not file_path:
            response = jsonify({'error': 'File path not provided'})
            response.status_code = 400
            return response
            
        if not os.path.exists(str(file_path)):
            response = jsonify({'error': 'File not found'})
            response.status_code = 400
            return response
            
        # Load and process volume
        volume, error = safe_load_volume(str(file_path))
        if error:
            response = jsonify({'error': f"Error loading MRI: {error}"})
            response.status_code = 400
            return response
        
        if volume is not None:
            volume, error = safe_normalize_volume(volume)
            if error:
                response = jsonify({'error': f"Error processing MRI: {error}"})
                response.status_code = 400
                return response
        else:
            response = jsonify({'error': "Failed to load volume data"})
            response.status_code = 400
            return response
        
        # Segment tumor
        if volume is not None:
            predicted_mask, error = safe_segment_tumor(volume, threshold, min_size)
            if error:
                response = jsonify({'error': f"Error in tumor detection: {error}"})
                response.status_code = 400
                return response
        else:
            response = jsonify({'error': "Failed to load volume data for segmentation"})
            response.status_code = 400
            return response
        
        # Calculate metrics
        tumor_volume = np.sum(predicted_mask) if predicted_mask is not None else 0
        total_volume = volume.shape[0] * volume.shape[1] * volume.shape[2] if volume is not None else 0
        tumor_percentage = (tumor_volume / total_volume) * 100 if total_volume > 0 else 0
        
        tumor_size = None
        tumor_center = None
        if tumor_volume > 0 and predicted_mask is not None:
            tumor_mask_bool = predicted_mask.astype(bool)
            tumor_coords = np.where(tumor_mask_bool)
            if tumor_coords[0].size > 0:
                tumor_size = [int(np.max(coord) - np.min(coord) + 1) for coord in tumor_coords]
                tumor_center = [float(np.mean(coord)) for coord in tumor_coords]
        
        confidence_score = (1 - threshold) * 100
        
        # Return results
        response = jsonify({
            'tumor_volume': float(tumor_volume),
            'tumor_percentage': float(tumor_percentage),
            'total_volume': float(total_volume),
            'tumor_size': tumor_size,
            'tumor_center': tumor_center,
            'confidence_score': float(confidence_score),
            'predicted_mask': predicted_mask.tolist() if predicted_mask is not None else None,
            'volume_shape': [int(x) for x in volume.shape] if volume is not None else None
        })
        response.status_code = 200
        return response
        
    except Exception as e:
        response = jsonify({'error': f"Error processing MRI: {str(e)}"})
        response.status_code = 500
        return response
    # Default response in case of unexpected flow
    response = jsonify({'error': 'Unexpected error'})
    response.status_code = 500
    return response

@app.route('/api/3d-visualization', methods=['POST'])
def get_3d_visualization() -> Response:
    """Generate 3D visualization data"""
    try:
        data = request.get_json()
        if not data:
            response = jsonify({'error': 'No JSON data provided'})
            response.status_code = 400
            return response
            
        file_path = data.get('file_path') if data else None
        threshold = float(data.get('threshold', 0.65)) if data else 0.65
        min_size = int(data.get('min_size', 100)) if data else 100
        brain_opacity = float(data.get('brain_opacity', 0.2)) if data else 0.2
        brain_color = data.get('brain_color', '#ADD8E6') if data else '#ADD8E6'
        tumor_opacity = float(data.get('tumor_opacity', 0.6)) if data else 0.6
        tumor_color = data.get('tumor_color', '#FF0000') if data else '#FF0000'
        
        if not file_path:
            response = jsonify({'error': 'File path not provided'})
            response.status_code = 400
            return response
            
        if not os.path.exists(str(file_path)):
            response = jsonify({'error': 'File not found'})
            response.status_code = 400
            return response
            
        # Load and process volume
        volume, error = safe_load_volume(str(file_path))
        if error:
            response = jsonify({'error': f"Error loading MRI: {error}"})
            response.status_code = 400
            return response
        
        if volume is not None:
            volume, error = safe_normalize_volume(volume)
            if error:
                response = jsonify({'error': f"Error processing MRI: {error}"})
                response.status_code = 400
                return response
        else:
            response = jsonify({'error': "Failed to load volume data"})
            response.status_code = 400
            return response
        
        # Segment tumor
        if volume is not None:
            predicted_mask, error = safe_segment_tumor(volume, threshold, min_size)
            if error:
                response = jsonify({'error': f"Error in tumor detection: {error}"})
                response.status_code = 400
                return response
        else:
            response = jsonify({'error': "Failed to load volume data for segmentation"})
            response.status_code = 400
            return response
        
        # Generate 3D data for brain
        brain_result, error = safe_marching_cubes(volume, 0.5)
        if error:
            response = jsonify({'error': error})
            response.status_code = 400
            return response
        
        if brain_result is not None:
            verts, faces, _, _ = brain_result
        else:
            response = jsonify({'error': "Failed to generate brain visualization"})
            response.status_code = 400
            return response
        
        # Prepare brain mesh data
        brain_data = {
            'x': verts[:, 0].tolist(),
            'y': verts[:, 1].tolist(),
            'z': verts[:, 2].tolist(),
            'i': faces[:, 0].tolist(),
            'j': faces[:, 1].tolist(),
            'k': faces[:, 2].tolist(),
            'color': brain_color,
            'opacity': brain_opacity,
            'name': 'Brain'
        }
        
        # Generate 3D data for tumor if present
        tumor_data = None
        if predicted_mask is not None and np.any(predicted_mask):
            tumor_result, error = safe_marching_cubes(predicted_mask, 0.5)
            if not error and tumor_result is not None:
                tumor_verts, tumor_faces, _, _ = tumor_result
                tumor_data = {
                    'x': tumor_verts[:, 0].tolist(),
                    'y': tumor_verts[:, 1].tolist(),
                    'z': tumor_verts[:, 2].tolist(),
                    'i': tumor_faces[:, 0].tolist(),
                    'j': tumor_faces[:, 1].tolist(),
                    'k': tumor_faces[:, 2].tolist(),
                    'color': tumor_color,
                    'opacity': tumor_opacity,
                    'name': 'Predicted Tumor'
                }
        
        response = jsonify({
            'brain_data': brain_data,
            'tumor_data': tumor_data
        })
        response.status_code = 200
        return response
        
    except Exception as e:
        response = jsonify({'error': f"Error generating 3D visualization: {str(e)}"})
        response.status_code = 500
        return response

@app.route('/api/slice', methods=['POST'])
def get_slice() -> Response:
    """Get a specific slice of the MRI"""
    try:
        data = request.get_json()
        print(f"Slice endpoint received data: {data}")  # Debug line
        
        if not data:
            response = jsonify({'error': 'No JSON data provided'})
            response.status_code = 400
            return response
            
        file_path = data.get('file_path') if data else None
        threshold = float(data.get('threshold', 0.65)) if data else 0.65
        min_size = int(data.get('min_size', 100)) if data else 100
        slice_idx = int(data.get('slice_idx', 0)) if data else 0
        contrast = float(data.get('contrast', 1.0)) if data else 1.0
        brightness = float(data.get('brightness', 0.0)) if data else 0.0
        show_mask = data.get('show_mask', True) if data else True
        
        print(f"Slice parameters: file_path={file_path}, threshold={threshold}, min_size={min_size}, slice_idx={slice_idx}, contrast={contrast}, brightness={brightness}, show_mask={show_mask}")  # Debug line
        
        if not file_path:
            response = jsonify({'error': 'File path not provided'})
            response.status_code = 400
            return response
            
        if not os.path.exists(str(file_path)):
            response = jsonify({'error': 'File not found'})
            response.status_code = 400
            return response
            
        # Load and process volume
        volume, error = safe_load_volume(str(file_path))
        if error:
            response = jsonify({'error': f"Error loading MRI: {error}"})
            response.status_code = 400
            return response
        
        if volume is not None:
            volume, error = safe_normalize_volume(volume)
            if error:
                response = jsonify({'error': f"Error processing MRI: {error}"})
                response.status_code = 400
                return response
        else:
            response = jsonify({'error': "Failed to load volume data"})
            response.status_code = 400
            return response
        
        # Segment tumor
        if volume is not None:
            predicted_mask, error = safe_segment_tumor(volume, threshold, min_size)
            if error:
                response = jsonify({'error': f"Error in tumor detection: {error}"})
                response.status_code = 400
                return response
        else:
            response = jsonify({'error': "Failed to load volume data for segmentation"})
            response.status_code = 400
            return response
        
        # Get slice
        print(f"Getting slice: volume.shape={volume.shape if volume is not None else None}, slice_idx={slice_idx}")  # Debug line
        if volume is not None and slice_idx >= volume.shape[2]:
            slice_idx = volume.shape[2] - 1
            print(f"Adjusted slice_idx to {slice_idx}")  # Debug line
            
        slice_img = volume[:, :, slice_idx] if volume is not None else np.array([])
        print(f"slice_img.shape={slice_img.shape if slice_img is not None and hasattr(slice_img, 'shape') else 'N/A'}")  # Debug line
        mask_slice = None
        if predicted_mask is not None and show_mask and predicted_mask.shape[2] > slice_idx:
            mask_slice = predicted_mask[:, :, slice_idx]
            print(f"mask_slice.shape={mask_slice.shape if mask_slice is not None and hasattr(mask_slice, 'shape') else 'N/A'}")  # Debug line
        
        # Visualize slice
        result = None
        error = None
        if slice_img is not None and slice_img.size > 0:
            print("Calling safe_visualize_slice")  # Debug line
            result, error = safe_visualize_slice(slice_img, mask_slice, contrast, brightness)
            print(f"safe_visualize_slice result: result={result is not None}, error={error}")  # Debug line
        else:
            error = "No slice data available"
            print(f"No slice data available: slice_img is None or empty")  # Debug line
            
        if error and result is None:
            print(f"Returning error: {error}")  # Debug line
            response = jsonify({'error': error})
            response.status_code = 400
            return response
        
        # Convert image to base64 for transmission
        print("Converting image to base64")  # Debug line
        import base64
        if result is not None and cv2 is not None:
            imencode_func = getattr(cv2, 'imencode', lambda x, y: (True, y))
            _, buffer = imencode_func('.png', result)
            img_str = base64.b64encode(buffer).decode()
            print("Image encoded successfully")  # Debug line
        else:
            img_str = ""
            print("No image to encode")  # Debug line
        
        response = jsonify({
            'image': img_str,
            'slice_idx': slice_idx
        })
        print("Returning successful response")  # Debug line
        response.status_code = 200
        return response
        
    except Exception as e:
        response = jsonify({'error': f"Error getting slice: {str(e)}"})
        response.status_code = 500
        return response

@app.route('/api/details', methods=['POST'])
def get_details() -> Response:
    """Get detailed analysis of the MRI"""
    try:
        data = request.get_json()
        if not data:
            response = jsonify({'error': 'No JSON data provided'})
            response.status_code = 400
            return response
            
        file_path = data.get('file_path') if data else None
        threshold = float(data.get('threshold', 0.65)) if data else 0.65
        min_size = int(data.get('min_size', 100)) if data else 100
        
        if not file_path:
            response = jsonify({'error': 'File path not provided'})
            response.status_code = 400
            return response
            
        if not os.path.exists(str(file_path)):
            response = jsonify({'error': 'File not found'})
            response.status_code = 400
            return response
            
        # Load and process volume
        volume, error = safe_load_volume(str(file_path))
        if error:
            response = jsonify({'error': f"Error loading MRI: {error}"})
            response.status_code = 400
            return response
        
        if volume is not None:
            volume, error = safe_normalize_volume(volume)
            if error:
                response = jsonify({'error': f"Error processing MRI: {error}"})
                response.status_code = 400
                return response
        else:
            response = jsonify({'error': "Failed to load volume data"})
            response.status_code = 400
            return response
        
        # Segment tumor
        if volume is not None:
            predicted_mask, error = safe_segment_tumor(volume, threshold, min_size)
            if error:
                response = jsonify({'error': f"Error in tumor detection: {error}"})
                response.status_code = 400
                return response
        else:
            response = jsonify({'error': "Failed to load volume data for segmentation"})
            response.status_code = 400
            return response
        
        # Calculate metrics
        tumor_volume = np.sum(predicted_mask) if predicted_mask is not None else 0
        total_volume = volume.shape[0] * volume.shape[1] * volume.shape[2] if volume is not None else 0
        tumor_percentage = (tumor_volume / total_volume) * 100 if total_volume > 0 else 0
        
        # Calculate statistics
        tumor_mask_bool = predicted_mask.astype(bool) if predicted_mask is not None else np.array([])
        num_regions = len(np.unique(predicted_mask)) - 1 if predicted_mask is not None else 0
        if num_regions < 0:
            num_regions = 0
        
        avg_intensity = 0
        max_intensity = 0
        if np.any(tumor_mask_bool) and volume is not None:
            avg_intensity = float(np.mean(volume[tumor_mask_bool]))
            max_intensity = float(np.max(volume[tumor_mask_bool]))
        
        tumor_size = None
        tumor_center = None
        if np.any(tumor_mask_bool):
            tumor_coords = np.where(tumor_mask_bool)
            tumor_size = [int(np.max(coord) - np.min(coord)) for coord in tumor_coords]
            tumor_center = [float(np.mean(coord)) for coord in tumor_coords]
        
        response = jsonify({
            'tumor_statistics': {
                'num_regions': num_regions,
                'avg_intensity': avg_intensity,
                'max_intensity': max_intensity
            },
            'volume_statistics': {
                'total_volume': float(total_volume),
                'tumor_volume': float(tumor_volume),
                'tumor_percentage': float(tumor_percentage)
            },
            'tumor_dimensions': tumor_size,
            'tumor_center': tumor_center,
            'hypotheses': hypotheses
        })
        response.status_code = 200
        return response
        
    except Exception as e:
        response = jsonify({'error': f"Error getting details: {str(e)}"})
        response.status_code = 500
        return response

@app.route('/api/summary', methods=['POST'])
def get_summary() -> Response:
    """Generate AI summary"""
    try:
        data = request.get_json()
        print(f"Summary endpoint received data: {data}")  # Debug line
        
        if not data:
            response = jsonify({'error': 'No JSON data provided'})
            response.status_code = 400
            return response
            
        tumor_volume = float(data.get('tumor_volume', 0)) if data else 0
        tumor_percentage = float(data.get('tumor_percentage', 0)) if data else 0
        total_volume = float(data.get('total_volume', 0)) if data else 0
        confidence_score = float(data.get('confidence_score', 0)) if data else 0
        
        print(f"Summary parameters: tumor_volume={tumor_volume}, tumor_percentage={tumor_percentage}, total_volume={total_volume}, confidence_score={confidence_score}")  # Debug line
        
        # Generate AI summary
        print("Calling safe_generate_ai_summary")  # Debug line
        summary, error = safe_generate_ai_summary(
            tumor_volume,
            tumor_percentage,
            total_volume,
            confidence_score
        )
        print(f"safe_generate_ai_summary result: summary={summary is not None}, error={error}")  # Debug line
        
        if error:
            print(f"Summary generation error: {error}")  # Debug line
            response = jsonify({'error': error})
            response.status_code = 400
            return response
            
        response = jsonify({
            'summary': summary
        })
        print("Returning successful summary response")  # Debug line
        response.status_code = 200
        return response
        
    except Exception as e:
        print(f"Summary endpoint exception: {str(e)}")  # Debug line
        response = jsonify({'error': f"Error generating summary: {str(e)}"})
        response.status_code = 500
        return response

@app.route('/static/<path:path>')
def send_static(path) -> Response:
    """Serve static files"""
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)