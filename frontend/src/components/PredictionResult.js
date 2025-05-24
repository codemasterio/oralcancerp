import React from 'react';

const PredictionResult = ({ result }) => {
  if (!result) return null;

  const { class_name, confidence, probability, inference_time, visualization_url } = result;
  const isCancer = class_name === 'Cancer';
  
  // Format confidence as percentage
  const confidencePercent = (confidence * 100).toFixed(2);
  
  return (
    <div className="results-section">
      <div className="result-header">
        <h2 className="result-title">Analysis Results</h2>
        <div>
          <span style={{ fontSize: '0.9rem', color: '#6c757d' }}>
            Processed in {inference_time ? inference_time.toFixed(2) : '0.00'} seconds
          </span>
        </div>
      </div>
      
      <div className="result-content">
        <div className="result-image">
          {visualization_url ? (
            <img 
              src={visualization_url} 
              alt="Analysis visualization" 
              className="result-visualization"
            />
          ) : (
            <div className="result-placeholder">
              <i className="fas fa-image" style={{ fontSize: '3rem', color: '#ced4da' }}></i>
              <p>Visualization not available</p>
            </div>
          )}
        </div>
        
        <div className="result-details">
          <div>
            <span 
              className={`prediction-label ${isCancer ? 'prediction-cancer' : 'prediction-non-cancer'}`}
            >
              <i className={`fas ${isCancer ? 'fa-exclamation-triangle' : 'fa-check-circle'} mr-2`}></i>
              {class_name}
            </span>
          </div>
          
          <div className="confidence-meter mt-4">
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>Confidence</span>
              <span><strong>{confidencePercent}%</strong></span>
            </div>
            <div className="confidence-bar">
              <div 
                className="confidence-fill" 
                style={{ 
                  width: `${confidencePercent}%`,
                  backgroundColor: isCancer ? '#dc3545' : '#28a745'
                }}
              ></div>
            </div>
          </div>
          
          <div className="mt-4">
            <h4>What does this mean?</h4>
            {isCancer ? (
              <p>
                The analysis indicates potential signs of oral cancer with {confidencePercent}% confidence. 
                This is not a definitive diagnosis. Please consult with a healthcare professional immediately 
                for a proper clinical evaluation.
              </p>
            ) : (
              <p>
                The analysis did not detect signs of oral cancer with {confidencePercent}% confidence. 
                However, regular check-ups with dental professionals are still recommended for oral health.
              </p>
            )}
          </div>
          
          <div className="disclaimer">
            <strong>Medical Disclaimer:</strong> This tool is designed for screening purposes only and 
            should not replace professional medical advice, diagnosis, or treatment. Always seek the 
            advice of your dentist, physician, or other qualified health provider with any questions 
            regarding a medical condition.
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionResult;
