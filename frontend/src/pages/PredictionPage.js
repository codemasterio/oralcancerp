import React, { useState } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';

import ImageUpload from '../components/ImageUpload';
import PredictionResult from '../components/PredictionResult';

const PredictionPage = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  
  const handleImageUpload = (file) => {
    setSelectedImage(file);
    // Reset prediction result when a new image is uploaded
    setPredictionResult(null);
  };
  
  const handleSubmit = async () => {
    if (!selectedImage) {
      toast.warning('Please upload an image first');
      return;
    }
    
    setIsLoading(true);
    setPredictionResult(null);
    
    // Create form data
    const formData = new FormData();
    formData.append('file', selectedImage);
    
    try {
      // Make API request to backend
      const response = await axios.post('/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      // Handle successful response
      setPredictionResult(response.data);
      
      // Show success toast based on prediction
      if (response.data.class_name === 'Cancer') {
        toast.error('Potential signs of oral cancer detected. Please consult a healthcare professional.');
      } else {
        toast.success('No signs of oral cancer detected.');
      }
      
    } catch (error) {
      console.error('Error making prediction:', error);
      
      // Handle error
      let errorMessage = 'An error occurred during prediction. Please try again.';
      
      if (error.response) {
        // Server responded with an error
        errorMessage = error.response.data.detail || errorMessage;
      } else if (error.request) {
        // No response received
        errorMessage = 'No response from server. Please check your connection.';
      }
      
      toast.error(errorMessage);
      
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="container prediction-container">
      <h1 className="mb-4">Oral Cancer Detection</h1>
      
      <div className="card mb-4">
        <h3>Instructions</h3>
        <p>
          Upload a clear, well-lit image of the oral cavity. The image should focus on the area of concern.
          Our AI model will analyze the image and provide a prediction on whether signs of oral cancer are detected.
        </p>
        <p>
          <strong>Note:</strong> This tool is for screening purposes only and does not replace professional medical advice.
        </p>
      </div>
      
      <ImageUpload 
        onImageUpload={handleImageUpload} 
        isLoading={isLoading} 
      />
      
      {selectedImage && !isLoading && !predictionResult && (
        <div className="text-center mt-4 mb-4">
          <button 
            className="button" 
            onClick={handleSubmit}
          >
            <i className="fas fa-microscope mr-2"></i> Analyze Image
          </button>
        </div>
      )}
      
      {isLoading && (
        <div className="spinner-container">
          <div className="spinner"></div>
          <p className="mt-3">Analyzing image. Please wait...</p>
        </div>
      )}
      
      {predictionResult && (
        <PredictionResult result={predictionResult} />
      )}
      
      <div className="card mt-5 mb-4">
        <h3>What to Look For</h3>
        <p>
          Common signs of oral cancer include:
        </p>
        <ul>
          <li>Persistent mouth sores that don't heal</li>
          <li>Persistent mouth pain</li>
          <li>A lump or thickening in the cheek</li>
          <li>White or red patches on the gums, tongue, tonsils, or lining of the mouth</li>
          <li>Difficulty swallowing or chewing</li>
          <li>Difficulty moving the jaw or tongue</li>
          <li>Numbness of the tongue or other area of the mouth</li>
        </ul>
        <p className="mt-3">
          If you notice any of these symptoms, please consult a healthcare professional immediately.
        </p>
      </div>
    </div>
  );
};

export default PredictionPage;
