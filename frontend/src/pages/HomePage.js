import React from 'react';
import { Link } from 'react-router-dom';

const HomePage = () => {
  return (
    <div className="container">
      <section className="hero">
        <h1>Oral Cancer Detection</h1>
        <p>
          This AI-powered tool helps identify potential signs of oral cancer from images of the oral cavity. 
          It's designed to assist healthcare professionals in early detection and monitoring. 
          The tool provides preliminary analysis but should never replace professional medical advice. 
          Always consult with your dentist or healthcare provider for any concerns about your oral health.
        </p>
        <Link to="/predict" className="button">
          <i className="fas fa-microscope mr-2"></i> Try It Now
        </Link>
      </section>
      
      <section>
        <h2 className="text-center mb-4">How It Works</h2>
        <div className="features">
          <div className="feature-card">
            <div className="feature-icon">
              <i className="fas fa-upload"></i>
            </div>
            <h3 className="feature-title">Upload</h3>
            <p>
              Upload a clear, well-lit image of the oral cavity. The image should focus on the area of concern.
            </p>
          </div>
          
          <div className="feature-card">
            <div className="feature-icon">
              <i className="fas fa-robot"></i>
            </div>
            <h3 className="feature-title">Analyze</h3>
            <p>
              Our advanced AI model analyzes the image using deep learning techniques trained on thousands of cases.
            </p>
          </div>
          
          <div className="feature-card">
            <div className="feature-icon">
              <i className="fas fa-chart-bar"></i>
            </div>
            <h3 className="feature-title">Results</h3>
            <p>
              Receive instant results with a confidence score and visualization highlighting areas of concern.
            </p>
          </div>
        </div>
      </section>
      
      <section className="mt-5">
        <div className="card">
          <h2 className="mb-3">Why Early Detection Matters</h2>
          <p>
            Oral cancer, when detected early, has a significantly higher survival rate. The 5-year survival rate for oral cancer 
            is approximately 84% when detected in its early stages, but drops to around 39% when detected in later stages.
          </p>
          <p>
            Regular screening and early detection are crucial for improving outcomes. Our tool aims to make screening more 
            accessible and help identify potential cases that require professional medical attention.
          </p>
          <div className="mt-3">
            <Link to="/about" className="button" style={{ backgroundColor: 'transparent', color: 'var(--primary-color)', border: '1px solid var(--primary-color)' }}>
              Learn More About Our Technology
            </Link>
          </div>
        </div>
      </section>
      
      <section className="mt-5 mb-5">
        <div className="card" style={{ backgroundColor: '#e3f2fd' }}>
          <h2 className="mb-3">Medical Disclaimer</h2>
          <p>
            This tool is designed for screening purposes only and should not replace professional medical advice, 
            diagnosis, or treatment. Always seek the advice of your dentist, physician, or other qualified health 
            provider with any questions regarding a medical condition.
          </p>
          <p>
            The results provided by this tool are not a definitive diagnosis and should be verified by healthcare professionals.
          </p>
        </div>
      </section>
    </div>
  );
};

export default HomePage;
