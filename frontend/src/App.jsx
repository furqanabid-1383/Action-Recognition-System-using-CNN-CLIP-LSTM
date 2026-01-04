import { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [apiStatus, setApiStatus] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [showActions, setShowActions] = useState(false);
  const [supportedActions, setSupportedActions] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  
  const fileInputRef = useRef(null);
  const dropZoneRef = useRef(null);

  const API_URL = 'http://localhost:8000';

  useEffect(() => {
    checkApiStatus();
    const interval = setInterval(checkApiStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const checkApiStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/health`, { method: 'GET' });
      setApiStatus(response.ok);
    } catch {
      setApiStatus(false);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setUploadedFile(file);
      const reader = new FileReader();
      reader.onload = (event) => setImagePreview(event.target.result);
      reader.readAsDataURL(file);
      setResult(null);
    }
  };

  // Drag and Drop handlers
  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (file.type.startsWith('image/')) {
        setUploadedFile(file);
        const reader = new FileReader();
        reader.onload = (event) => setImagePreview(event.target.result);
        reader.readAsDataURL(file);
        setResult(null);
      }
    }
  };

  const handleRemoveFile = () => {
    setUploadedFile(null);
    setImagePreview(null);
    setResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const predictImage = async () => {
    if (!uploadedFile || !apiStatus) return;

    setAnalyzing(true);
    const formData = new FormData();
    formData.append('file', uploadedFile);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setResult(data);
        } else {
          alert('Failed to process image. Please try again.');
        }
      } else {
        alert('Failed to process image. Please try again.');
      }
    } catch (error) {
      alert('Backend server is not running. Please start the API first.');
    } finally {
      setAnalyzing(false);
    }
  };

  const loadSupportedActions = async () => {
    try {
      const response = await fetch(`${API_URL}/actions`);
      if (response.ok) {
        const data = await response.json();
        setSupportedActions(data.actions || []);
      }
    } catch {
      console.log('Failed to load actions');
    }
  };

  const toggleActions = () => {
    if (!showActions && supportedActions.length === 0) {
      loadSupportedActions();
    }
    setShowActions(!showActions);
  };

  const downloadResults = () => {
    if (!result) return;
    const jsonStr = JSON.stringify(result, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'action_recognition_results.json';
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleUploadNewImage = () => {
    setUploadedFile(null);
    setImagePreview(null);
    setResult(null);
    
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getConfidenceGradient = (confidence) => {
    if (confidence >= 0.7) return 'linear-gradient(90deg, #00ff88 0%, #00cc6a 100%)';
    if (confidence >= 0.4) return 'linear-gradient(90deg, #ffd700 0%, #ffaa00 100%)';
    return 'linear-gradient(90deg, #ff4757 0%, #ff2f40 100%)';
  };

  return (
    <div className="app">
      {/* Header */}
      <div className="custom-header">
        <h1>Action Recognition AI</h1>
        <p>Advanced deep learning powered action detection and image understanding</p>
      </div>

      {/* Status Bar */}
      <div className="status-bar">
        <div className="status-item">
          <span className={apiStatus ? 'status-connected' : 'status-disconnected'}>●</span>
          <span>Backend {apiStatus ? 'Connected' : 'Offline'}</span>
        </div>
        <div className="status-item">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ color: 'rgba(255,255,255,0.6)' }}>
            <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z" />
            <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z" />
          </svg>
          <span style={{ color: 'rgba(255,255,255,0.8)' }}>CLIP ViT-B/32</span>
        </div>
        <div className="status-item">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ color: 'rgba(255,255,255,0.6)' }}>
            <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
          </svg>
          <span style={{ color: 'rgba(255,255,255,0.8)' }}>Version 2.0</span>
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        {/* Left Column - Upload */}
        <div className="column">
          <h3 className="section-header">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: 'middle', marginRight: '0.5rem' }}>
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            Upload Image
          </h3>

          <div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/jpg,image/png"
              onChange={handleFileChange}
              style={{ display: 'none' }}
              id="file-input"
            />

            {/* Drag and Drop Area */}
            <div
              ref={dropZoneRef}
              className={`drag-drop-zone ${isDragging ? 'dragging' : ''}`}
              onDragEnter={handleDragEnter}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <div className="drag-drop-content">
                <svg className="upload-icon" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
                <h4>Drag and drop file here</h4>
                <p className="drag-drop-limit">Limit 200MB per file • JPG, JPEG, PNG</p>
              </div>
              <button 
                className="browse-button"
                onClick={() => fileInputRef.current?.click()}
              >
                Browse files
              </button>
            </div>

            {/* File Preview */}
            {uploadedFile && (
              <div className="file-preview">
                <div className="file-info">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" />
                    <polyline points="13 2 13 9 20 9" />
                  </svg>
                  <div className="file-details">
                    <span className="file-name">{uploadedFile.name}</span>
                    <span className="file-size">
                      {uploadedFile.size ? (uploadedFile.size / 1024).toFixed(1) : 0}KB
                    </span>
                  </div>
                  <button className="remove-file-btn" onClick={handleRemoveFile}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <line x1="18" y1="6" x2="6" y2="18" />
                      <line x1="6" y1="6" x2="18" y2="18" />
                    </svg>
                  </button>
                </div>

                {imagePreview && (
                  <div className="image-preview-container">
                    <img src={imagePreview} alt="Preview" className="uploaded-image" />
                  </div>
                )}
              </div>
            )}

            {/* Action Buttons */}
            {uploadedFile && !result && (
              <button 
                className="analyze-button" 
                onClick={predictImage} 
                disabled={analyzing || !apiStatus}
                style={{ marginTop: '1.5rem' }}
              >
                {analyzing ? (
                  <>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="spinning-icon" style={{ verticalAlign: 'middle', marginRight: '0.5rem' }}>
                      <line x1="12" y1="2" x2="12" y2="6" />
                      <line x1="12" y1="18" x2="12" y2="22" />
                      <line x1="4.93" y1="4.93" x2="7.76" y2="7.76" />
                      <line x1="16.24" y1="16.24" x2="19.07" y2="19.07" />
                      <line x1="2" y1="12" x2="6" y2="12" />
                      <line x1="18" y1="12" x2="22" y2="12" />
                      <line x1="4.93" y1="19.07" x2="7.76" y2="16.24" />
                      <line x1="16.24" y1="7.76" x2="19.07" y2="4.93" />
                    </svg>
                    Processing with AI...
                  </>
                ) : (
                  <>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: 'middle', marginRight: '0.5rem' }}>
                      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
                    </svg>
                    Analyze Image
                  </>
                )}
              </button>
            )}
          </div>
        </div>

        {/* Right Column - Results */}
        <div className="column">
          <h3 className="section-header">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: 'middle', marginRight: '0.5rem' }}>
              <line x1="18" y1="20" x2="18" y2="10" />
              <line x1="12" y1="20" x2="12" y2="4" />
              <line x1="6" y1="20" x2="6" y2="14" />
            </svg>
            Analysis Results
          </h3>

          {result ? (
            <div>
              {/* Main Prediction */}
              <div className="result-card">
                <div className="prediction-badge">
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: 'middle', marginRight: '0.75rem' }}>
                    <circle cx="12" cy="12" r="10" />
                    <circle cx="12" cy="12" r="6" />
                    <circle cx="12" cy="12" r="2" />
                  </svg>
                  {result.predictions.action.toUpperCase()}
                </div>

                {/* Confidence Meter */}
                <div className="confidence-meter">
                  <p className="confidence-label">Confidence Level</p>
                  <div className="confidence-bar-container">
                    <div
                      className="confidence-bar"
                      style={{
                        width: `${result.predictions.confidence * 100}%`,
                        background: getConfidenceGradient(result.predictions.confidence),
                      }}
                    >
                      {(result.predictions.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>

              {/* Caption */}
              {result.predictions.caption && (
                <div className="caption-box">
                  {result.predictions.caption}
                </div>
              )}

              <div className="custom-divider"></div>

              {/* Top Predictions */}
              <div className="result-card">
                <h4 style={{ color: 'white', marginTop: 0, fontSize: '1.3rem', fontWeight: 700, marginBottom: '1.5rem' }}>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: 'middle', marginRight: '0.5rem' }}>
                    <circle cx="12" cy="12" r="10" />
                    <circle cx="12" cy="12" r="6" />
                    <circle cx="12" cy="12" r="2" />
                  </svg>
                  Alternative Predictions
                </h4>
                {result.predictions.top_predictions?.slice(0, 5).map((pred, idx) => (
                  <div key={idx} className="action-item-with-bar">
                    <div className="action-header">
                      <span className="action-name">{pred.action.charAt(0).toUpperCase() + pred.action.slice(1)}</span>
                      <span className="action-confidence-percentage">{(pred.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="action-progress-bar">
                      <div 
                        className="action-progress-fill"
                        style={{
                          width: `${pred.confidence * 100}%`,
                          background: getConfidenceGradient(pred.confidence)
                        }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Download Button */}
              <br />
              <button className="download-button" onClick={downloadResults}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: 'middle', marginRight: '0.5rem' }}>
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="7 10 12 15 17 10" />
                  <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
                Download Results (JSON)
              </button>
            </div>
          ) : (
            <div>
              <div className="info-box">
                <h4>No results yet</h4>
                <p style={{ marginBottom: 0, fontSize: '1.05rem' }}>
                  Upload an image and click "Analyze Image" to see AI-powered predictions and insights.
                </p>
              </div>

              <br />

              {/* Show Supported Actions */}
              <div className="expander">
                <button className="expander-header" onClick={toggleActions}>
                  {showActions ? '▼' : '▶'} View Supported Actions
                </button>
                {showActions && (
                  <div className="expander-content">
                    {supportedActions.length > 0 ? (
                      <div className="actions-grid">
                        {supportedActions.map((action, idx) => (
                          <p key={idx} style={{ color: 'rgba(255,255,255,0.8)', margin: '0.3rem 0' }}>
                            • {action.charAt(0).toUpperCase() + action.slice(1)}
                          </p>
                        ))}
                      </div>
                    ) : (
                      <p style={{ color: 'rgba(255,255,255,0.8)' }}>Backend not available</p>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="custom-divider"></div>

      <div className="footer">
        <div className="footer-section">
          <h4>Technology Stack</h4>
          <p>Frontend: React + Vite</p>
          <p>Backend: FastAPI</p>
          <p>Model: OpenAI CLIP</p>
        </div>
        <div className="footer-section">
          <h4>Features</h4>
          <p>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: 'middle', marginRight: '0.5rem' }}>
              <circle cx="12" cy="12" r="10" />
              <circle cx="12" cy="12" r="6" />
              <circle cx="12" cy="12" r="2" />
            </svg>
            Zero-shot learning
          </p>
          <p>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: 'middle', marginRight: '0.5rem' }}>
              <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
            </svg>
            Real-time prediction
          </p>
          <p>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: 'middle', marginRight: '0.5rem' }}>
              <rect x="2" y="7" width="20" height="14" rx="2" ry="2" />
              <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16" />
            </svg>
            REST API integration
          </p>
        </div>
        <div className="footer-section">
          <h4>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: 'middle', marginRight: '0.5rem' }}>
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="16" x2="12" y2="12" />
              <line x1="12" y1="8" x2="12.01" y2="8" />
            </svg>
            About
          </h4>
          <p>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: 'middle', marginRight: '0.5rem' }}>
              <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z" />
              <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z" />
            </svg>
            Deep Learning Project
          </p>
          <p>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: 'middle', marginRight: '0.5rem' }}>
              <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
              <circle cx="12" cy="12" r="3" />
            </svg>
            Action Recognition
          </p>
          <p>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: 'middle', marginRight: '0.5rem' }}>
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
            Image Captioning
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;