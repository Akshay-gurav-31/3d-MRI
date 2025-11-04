// Global variables
let currentFilePath = null;
let currentVolumeShape = null;
let currentResults = null;
let currentTheme = 'light';

// DOM Elements
const uploadBtn = document.getElementById('uploadBtn');
const mriFile = document.getElementById('mriFile');
const threshold = document.getElementById('threshold');
const minSize = document.getElementById('minSize');
const thresholdValue = document.getElementById('thresholdValue');
const minSizeValue = document.getElementById('minSizeValue');
const themeToggle = document.getElementById('themeToggle');

// 3D Visualization controls
const brainOpacity = document.getElementById('brainOpacity');
const brainColor = document.getElementById('brainColor');
const tumorOpacity = document.getElementById('tumorOpacity');
const tumorColor = document.getElementById('tumorColor');
const showAxes = document.getElementById('showAxes');
const showLegend = document.getElementById('showLegend');
const brainOpacityValue = document.getElementById('brainOpacityValue');
const tumorOpacityValue = document.getElementById('tumorOpacityValue');

// Slice controls
const sliceIndex = document.getElementById('sliceIndex');
const contrast = document.getElementById('contrast');
const brightness = document.getElementById('brightness');
const showMask = document.getElementById('showMask');
const sliceIndexValue = document.getElementById('sliceIndexValue');
const maxSliceIndex = document.getElementById('maxSliceIndex');
const contrastValue = document.getElementById('contrastValue');
const brightnessValue = document.getElementById('brightnessValue');

// Metric elements
const tumorVolume = document.getElementById('tumorVolume');
const tumorPercentage = document.getElementById('tumorPercentage');
const confidenceScore = document.getElementById('confidenceScore');

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    // Initialize slider value displays
    updateSliderValues();
    
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        setTheme(savedTheme);
    }
    
    // Add event listeners for sliders
    threshold.addEventListener('input', function() {
        thresholdValue.textContent = this.value;
    });
    
    minSize.addEventListener('input', function() {
        minSizeValue.textContent = this.value;
    });
    
    brainOpacity.addEventListener('input', function() {
        brainOpacityValue.textContent = this.value;
    });
    
    tumorOpacity.addEventListener('input', function() {
        tumorOpacityValue.textContent = this.value;
    });
    
    contrast.addEventListener('input', function() {
        contrastValue.textContent = this.value;
    });
    
    brightness.addEventListener('input', function() {
        brightnessValue.textContent = this.value;
    });
    
    // Upload button event
    uploadBtn.addEventListener('click', handleFileUpload);
    
    // Theme toggle event
    themeToggle.addEventListener('click', toggleTheme);
    
    // Tab change events
    document.getElementById('visualization-tab').addEventListener('click', load3DVisualization);
    document.getElementById('slices-tab').addEventListener('click', loadSlices);
    document.getElementById('details-tab').addEventListener('click', loadDetails);
    document.getElementById('summary-tab').addEventListener('click', loadSummary);
    
    // 3D visualization controls
    brainOpacity.addEventListener('change', load3DVisualization);
    brainColor.addEventListener('change', load3DVisualization);
    tumorOpacity.addEventListener('change', load3DVisualization);
    tumorColor.addEventListener('change', load3DVisualization);
    showAxes.addEventListener('change', load3DVisualization);
    showLegend.addEventListener('change', load3DVisualization);
    
    // Slice controls
    sliceIndex.addEventListener('input', function() {
        sliceIndexValue.textContent = this.value;
    });
    
    sliceIndex.addEventListener('change', loadSlices);
    contrast.addEventListener('change', loadSlices);
    brightness.addEventListener('change', loadSlices);
    showMask.addEventListener('change', loadSlices);
});

// Update slider value displays
function updateSliderValues() {
    thresholdValue.textContent = threshold.value;
    minSizeValue.textContent = minSize.value;
    brainOpacityValue.textContent = brainOpacity.value;
    tumorOpacityValue.textContent = tumorOpacity.value;
    sliceIndexValue.textContent = sliceIndex.value;
    contrastValue.textContent = contrast.value;
    brightnessValue.textContent = brightness.value;
}

// Toggle theme
function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-bs-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
}

// Set theme
function setTheme(theme) {
    document.documentElement.setAttribute('data-bs-theme', theme);
    currentTheme = theme;
    localStorage.setItem('theme', theme);
    
    // Update theme toggle button icon
    const themeIcon = themeToggle.querySelector('i');
    if (theme === 'dark') {
        themeIcon.className = 'fas fa-sun';
    } else {
        themeIcon.className = 'fas fa-moon';
    }
    
    // Re-render Plotly if it exists
    if (document.getElementById('plotly-3d-container') && 
        document.getElementById('plotly-3d-container').data) {
        load3DVisualization();
    }
    
    // Update 3D container background color directly
    const plotlyContainer = document.getElementById('plotly-3d-container');
    if (plotlyContainer) {
        plotlyContainer.style.backgroundColor = theme === 'dark' ? '#000000' : '#f8f9fa';
    }
}

// Show status message
function showStatus(message, type = 'info') {
    const statusMessages = document.getElementById('statusMessages');
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.role = 'alert';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    statusMessages.appendChild(alertDiv);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.parentNode.removeChild(alertDiv);
        }
    }, 5000);
}

// Handle file upload
async function handleFileUpload() {
    const file = mriFile.files[0];
    if (!file) {
        showStatus('Please select a file to upload.', 'warning');
        return;
    }
    
    showStatus('Uploading file...', 'info');
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            currentFilePath = result.file_path;
            showStatus('File uploaded successfully. Processing...', 'success');
            
            // Process the MRI
            await processMRI();
        } else {
            showStatus(`Upload failed: ${result.error}`, 'danger');
        }
    } catch (error) {
        showStatus(`Upload error: ${error.message}`, 'danger');
    }
}

// Process MRI
async function processMRI() {
    if (!currentFilePath) return;
    
    showStatus('Processing MRI...', 'info');
    
    try {
        const response = await fetch('/api/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                file_path: currentFilePath,
                threshold: parseFloat(threshold.value),
                min_size: parseInt(minSize.value)
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            currentResults = result;
            currentVolumeShape = result.volume_shape;
            
            // Update metrics
            tumorVolume.textContent = result.tumor_volume.toLocaleString();
            tumorPercentage.textContent = result.tumor_percentage.toFixed(2) + '%';
            confidenceScore.textContent = result.confidence_score.toFixed(1) + '%';
            
            // Update slice index max value
            if (currentVolumeShape && currentVolumeShape.length >= 3) {
                maxSliceIndex.textContent = currentVolumeShape[2] - 1;
                sliceIndex.max = currentVolumeShape[2] - 1;
                sliceIndex.value = Math.floor(currentVolumeShape[2] / 2);
                sliceIndexValue.textContent = sliceIndex.value;
            }
            
            showStatus('MRI processed successfully!', 'success');
            
            // Load the 3D visualization by default
            load3DVisualization();
        } else {
            showStatus(`Processing failed: ${result.error}`, 'danger');
        }
    } catch (error) {
        showStatus(`Processing error: ${error.message}`, 'danger');
    }
}

// Load 3D visualization
async function load3DVisualization() {
    if (!currentFilePath || !currentResults) return;
    
    showStatus('Loading 3D visualization...', 'info');
    
    try {
        const response = await fetch('/api/3d-visualization', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                file_path: currentFilePath,
                threshold: parseFloat(threshold.value),
                min_size: parseInt(minSize.value),
                brain_opacity: parseFloat(brainOpacity.value),
                brain_color: brainColor.value,
                tumor_opacity: parseFloat(tumorOpacity.value),
                tumor_color: tumorColor.value
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            const brainData = result.brain_data;
            const tumorData = result.tumor_data;
            
            // Create brain mesh
            const brainMesh = {
                type: 'mesh3d',
                x: brainData.x,
                y: brainData.y,
                z: brainData.z,
                i: brainData.i,
                j: brainData.j,
                k: brainData.k,
                color: brainData.color,
                opacity: brainData.opacity,
                name: brainData.name,
                showscale: false
            };
            
            const data = [brainMesh];
            
            // Add tumor mesh if available
            if (tumorData) {
                const tumorMesh = {
                    type: 'mesh3d',
                    x: tumorData.x,
                    y: tumorData.y,
                    z: tumorData.z,
                    i: tumorData.i,
                    j: tumorData.j,
                    k: tumorData.k,
                    color: tumorData.color,
                    opacity: tumorData.opacity,
                    name: tumorData.name,
                    showscale: false
                };
                data.push(tumorMesh);
            }
            
            const layout = {
                scene: {
                    xaxis: {
                        title: 'Right-Left',
                        visible: showAxes.checked
                    },
                    yaxis: {
                        title: 'Anterior-Posterior',
                        visible: showAxes.checked
                    },
                    zaxis: {
                        title: 'Superior-Inferior',
                        visible: showAxes.checked
                    },
                    aspectmode: 'data',
                    camera: {
                        up: {x: 0, y: 0, z: 1},
                        center: {x: 0, y: 0, z: 0},
                        eye: {x: 1.5, y: 1.5, z: 1.5}
                    },
                    bgcolor: 'black' // Always black for better 3D visualization
                },
                margin: {l: 0, r: 0, b: 0, t: 0},
                showlegend: showLegend.checked,
                height: 500,
                dragmode: 'orbit',
                hovermode: 'closest',
                paper_bgcolor: 'black', // Background color for the entire plot
                plot_bgcolor: 'black'   // Background color for the plotting area
            };
            
            const config = {
                displayModeBar: true,
                scrollZoom: true,
                displaylogo: false,
                modeBarButtonsToAdd: ['orbitRotation', 'resetCamera'],
                modeBarButtonsToRemove: ['lasso2d', 'select2d']
            };
            
            Plotly.newPlot('plotly-3d-container', data, layout, config);
            showStatus('3D visualization loaded successfully!', 'success');
        } else {
            showStatus(`3D visualization failed: ${result.error}`, 'danger');
        }
    } catch (error) {
        showStatus(`3D visualization error: ${error.message}`, 'danger');
    }
}

// Load slices
async function loadSlices() {
    if (!currentFilePath || !currentResults) return;
    
    showStatus('Loading slice...', 'info');
    
    try {
        const response = await fetch('/api/slice', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                file_path: currentFilePath,
                threshold: parseFloat(threshold.value),
                min_size: parseInt(minSize.value),
                slice_idx: parseInt(sliceIndex.value),
                contrast: parseFloat(contrast.value),
                brightness: parseFloat(brightness.value),
                show_mask: showMask.checked
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            const sliceImage = document.getElementById('sliceImage');
            sliceImage.src = `data:image/png;base64,${result.image}`;
            showStatus('Slice loaded successfully!', 'success');
        } else {
            showStatus(`Slice loading failed: ${result.error}`, 'danger');
        }
    } catch (error) {
        showStatus(`Slice loading error: ${error.message}`, 'danger');
    }
}

// Load details
async function loadDetails() {
    if (!currentFilePath || !currentResults) return;
    
    showStatus('Loading details...', 'info');
    
    try {
        const response = await fetch('/api/details', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                file_path: currentFilePath,
                threshold: parseFloat(threshold.value),
                min_size: parseInt(minSize.value)
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            const detailsContent = document.getElementById('detailsContent');
            
            let html = `
                <h5>Tumor Statistics</h5>
                <ul>
                    <li>Number of tumor regions: ${result.tumor_statistics.num_regions}</li>
                    <li>Average tumor intensity: ${result.tumor_statistics.avg_intensity.toFixed(3)}</li>
                    <li>Maximum tumor intensity: ${result.tumor_statistics.max_intensity.toFixed(3)}</li>
                </ul>
                
                <h5>Volume Statistics</h5>
                <ul>
                    <li>Total volume: ${result.volume_statistics.total_volume.toLocaleString()} voxels</li>
                    <li>Tumor volume: ${result.volume_statistics.tumor_volume.toLocaleString()} voxels</li>
                    <li>Tumor percentage: ${result.volume_statistics.tumor_percentage.toFixed(2)}%</li>
                </ul>
            `;
            
            if (result.tumor_dimensions) {
                html += `
                    <h5>Tumor Dimensions (voxels)</h5>
                    <ul>
                        <li>X: ${result.tumor_dimensions[0]}</li>
                        <li>Y: ${result.tumor_dimensions[1]}</li>
                        <li>Z: ${result.tumor_dimensions[2]}</li>
                    </ul>
                `;
            }
            
            if (result.tumor_center) {
                html += `
                    <h5>Tumor Center (voxels)</h5>
                    <ul>
                        <li>X: ${result.tumor_center[0].toFixed(1)}</li>
                        <li>Y: ${result.tumor_center[1].toFixed(1)}</li>
                        <li>Z: ${result.tumor_center[2].toFixed(1)}</li>
                    </ul>
                `;
            }
            
            if (result.hypotheses) {
                html += `
                    <h5>Initial Hypotheses</h5>
                    <pre>${result.hypotheses}</pre>
                `;
            }
            
            detailsContent.innerHTML = html;
            showStatus('Details loaded successfully!', 'success');
        } else {
            showStatus(`Details loading failed: ${result.error}`, 'danger');
        }
    } catch (error) {
        showStatus(`Details loading error: ${error.message}`, 'danger');
    }
}

// Load summary
async function loadSummary() {
    if (!currentResults) return;
    
    showStatus('Generating AI summary...', 'info');
    
    try {
        const response = await fetch('/api/summary', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                tumor_volume: currentResults.tumor_volume,
                tumor_percentage: currentResults.tumor_percentage,
                total_volume: currentResults.total_volume,
                confidence_score: currentResults.confidence_score
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            const summaryContent = document.getElementById('summaryContent');
            summaryContent.innerHTML = `
                <h5>AI-Generated Medical Summary</h5>
                <div class="p-3 rounded" style="background-color: var(--metric-bg); border: 1px solid var(--border-color);">
                    ${result.summary}
                </div>
                <div class="alert alert-warning mt-3">
                    <strong>Disclaimer:</strong> This AI-generated summary is for informational purposes only and should not be considered as medical advice. Please consult with a qualified healthcare professional for proper medical interpretation and guidance.
                </div>
            `;
            showStatus('AI summary generated successfully!', 'success');
        } else {
            showStatus(`Summary generation failed: ${result.error}`, 'danger');
        }
    } catch (error) {
        showStatus(`Summary generation error: ${error.message}`, 'danger');
    }
}