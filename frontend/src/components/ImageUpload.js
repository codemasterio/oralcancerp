import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { toast } from 'react-toastify';

const ImageUpload = ({ onImageUpload, isLoading }) => {
  const [previewImages, setPreviewImages] = useState([]);

  const onDrop = useCallback((acceptedFiles) => {
    // Check if files are images
    const imageFiles = acceptedFiles.filter(file => 
      file.type.startsWith('image/')
    );
    
    if (imageFiles.length === 0) {
      toast.error('Please upload valid image files (JPEG, PNG)');
      return;
    }
    
    // Check file size (max 5MB)
    const maxSize = 5 * 1024 * 1024; // 5MB
    const validSizeFiles = imageFiles.filter(file => file.size <= maxSize);
    
    if (validSizeFiles.length < imageFiles.length) {
      toast.error('Some files exceed the maximum size of 5MB');
    }
    
    if (validSizeFiles.length === 0) {
      return;
    }
    
    // Create preview URLs
    const newPreviewImages = validSizeFiles.map(file => ({
      file,
      preview: URL.createObjectURL(file)
    }));
    
    setPreviewImages(newPreviewImages);
    
    // Pass the first valid file to parent component
    if (validSizeFiles.length > 0) {
      onImageUpload(validSizeFiles[0]);
    }
  }, [onImageUpload]);
  
  const removeImage = (index) => {
    const newPreviewImages = [...previewImages];
    
    // Revoke the object URL to avoid memory leaks
    URL.revokeObjectURL(newPreviewImages[index].preview);
    
    newPreviewImages.splice(index, 1);
    setPreviewImages(newPreviewImages);
    
    if (newPreviewImages.length > 0) {
      onImageUpload(newPreviewImages[0].file);
    } else {
      onImageUpload(null);
    }
  };
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png']
    },
    maxFiles: 1,
    disabled: isLoading
  });
  
  // Clean up preview URLs when component unmounts
  React.useEffect(() => {
    return () => {
      previewImages.forEach(image => {
        URL.revokeObjectURL(image.preview);
      });
    };
  }, [previewImages]);
  
  return (
    <div className="upload-section">
      <h2>Upload Oral Image</h2>
      <p className="mb-3">
        Upload a clear image of the oral cavity for cancer detection. 
        The image should be well-lit and focused.
      </p>
      
      <div 
        {...getRootProps()} 
        className={`dropzone ${isDragActive ? 'dropzone-active' : ''} ${isLoading ? 'opacity-50' : ''}`}
      >
        <input {...getInputProps()} />
        {isDragActive ? (
          <p>Drop the image here...</p>
        ) : (
          <div>
            <i className="fas fa-cloud-upload-alt" style={{ fontSize: '3rem', color: '#3498db', marginBottom: '15px' }}></i>
            <p>Drag & drop an image here, or click to select</p>
            <p className="mt-2" style={{ fontSize: '0.9rem', color: '#6c757d' }}>
              Supported formats: JPEG, PNG (Max size: 5MB)
            </p>
          </div>
        )}
      </div>
      
      {previewImages.length > 0 && (
        <div className="preview-container mt-3">
          {previewImages.map((image, index) => (
            <div key={index} className="image-preview">
              <img src={image.preview} alt={`Preview ${index}`} />
              {!isLoading && (
                <div 
                  className="remove-image" 
                  onClick={(e) => {
                    e.stopPropagation();
                    removeImage(index);
                  }}
                >
                  <i className="fas fa-times"></i>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
      
      {previewImages.length > 0 && (
        <div className="mt-3">
          <button 
            className="button" 
            onClick={() => onImageUpload(previewImages[0].file)}
            disabled={isLoading}
          >
            {isLoading ? (
              <>
                <i className="fas fa-spinner fa-spin mr-2"></i> Processing...
              </>
            ) : (
              <>
                <i className="fas fa-microscope mr-2"></i> Analyze Image
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
};

export default ImageUpload;
