import React from 'react';

const AboutPage = () => {
  return (
    <div className="container about-section">
      <h1 className="mb-4">About Our Technology</h1>
      
      <section className="card mb-4">
        <h2>Our Mission</h2>
        <p>
          Our mission is to improve early detection of oral cancer through accessible AI-powered screening tools.
          By leveraging advanced machine learning techniques, we aim to provide a supplementary tool for healthcare
          professionals and individuals to identify potential signs of oral cancer at an early stage.
        </p>
      </section>
      
      <section className="card mb-4">
        <h2>The Technology</h2>
        <p>
          Our oral cancer detection system uses state-of-the-art deep learning models trained on thousands of
          medical images. The technology behind our application includes:
        </p>
        <ul className="mt-3">
          <li>
            <strong>Convolutional Neural Networks (CNNs):</strong> We utilize advanced CNN architectures like 
            ResNet, EfficientNet, and DenseNet that have been fine-tuned specifically for oral cancer detection.
          </li>
          <li>
            <strong>Transfer Learning:</strong> Our models leverage pre-trained weights from large-scale image 
            datasets, which are then fine-tuned on specialized oral cancer image collections.
          </li>
          <li>
            <strong>Image Processing:</strong> Advanced preprocessing techniques ensure that uploaded images 
            are standardized for optimal analysis by our AI models.
          </li>
          <li>
            <strong>Visualization Techniques:</strong> We implement Grad-CAM (Gradient-weighted Class Activation Mapping) 
            to highlight regions of interest in the image that influenced the model's prediction.
          </li>
        </ul>
      </section>
      
      <section className="card mb-4">
        <h2>Performance Metrics</h2>
        <p>
          Our model has been evaluated on diverse datasets and has achieved the following performance metrics:
        </p>
        <div className="row mt-3">
          <div className="col-md-6 col-sm-12 mb-3">
            <div style={{ padding: '15px', backgroundColor: '#e3f2fd', borderRadius: '4px', textAlign: 'center' }}>
              <h3 style={{ fontSize: '1.2rem' }}>Accuracy</h3>
              <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#2c3e50' }}>92%</div>
            </div>
          </div>
          <div className="col-md-6 col-sm-12 mb-3">
            <div style={{ padding: '15px', backgroundColor: '#e3f2fd', borderRadius: '4px', textAlign: 'center' }}>
              <h3 style={{ fontSize: '1.2rem' }}>Sensitivity</h3>
              <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#2c3e50' }}>95%</div>
            </div>
          </div>
          <div className="col-md-6 col-sm-12 mb-3">
            <div style={{ padding: '15px', backgroundColor: '#e3f2fd', borderRadius: '4px', textAlign: 'center' }}>
              <h3 style={{ fontSize: '1.2rem' }}>Specificity</h3>
              <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#2c3e50' }}>88%</div>
            </div>
          </div>
          <div className="col-md-6 col-sm-12 mb-3">
            <div style={{ padding: '15px', backgroundColor: '#e3f2fd', borderRadius: '4px', textAlign: 'center' }}>
              <h3 style={{ fontSize: '1.2rem' }}>F1-Score</h3>
              <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#2c3e50' }}>93%</div>
            </div>
          </div>
        </div>
        <p>
          <strong>Note:</strong> These metrics are based on our validation datasets and may vary in real-world scenarios.
          The tool is continuously being improved as more data becomes available.
        </p>
      </section>
      
      <section className="card mb-4">
        <h2>Limitations</h2>
        <p>
          While our technology provides valuable screening capabilities, it's important to understand its limitations:
        </p>
        <ul className="mt-3">
          <li>
            This tool is not a replacement for professional medical diagnosis. All results should be confirmed by healthcare professionals.
          </li>
          <li>
            The accuracy of predictions depends on the quality of the uploaded images. Poor lighting, blurry images, or improper angles may affect results.
          </li>
          <li>
            The model may not detect all types of oral cancer with equal accuracy, particularly rare variants or those in very early stages.
          </li>
          <li>
            Cultural and demographic biases may exist in the training data, potentially affecting performance across different populations.
          </li>
        </ul>
      </section>
      
      <section className="card mb-4">
        <h2>Research and Development</h2>
        <p>
          Our technology is built on extensive research in medical imaging and artificial intelligence. We continuously 
          improve our models through:
        </p>
        <ul className="mt-3">
          <li>Collaboration with medical professionals and research institutions</li>
          <li>Incorporation of feedback from users and healthcare providers</li>
          <li>Regular retraining with expanded and diverse datasets</li>
          <li>Implementation of the latest advancements in AI and computer vision</li>
        </ul>
      </section>
      
      <section className="card mb-5">
        <h2>Privacy and Security</h2>
        <p>
          We take your privacy seriously. Our application:
        </p>
        <ul className="mt-3">
          <li>Does not permanently store uploaded images</li>
          <li>Processes all data securely</li>
          <li>Does not share your information with third parties</li>
          <li>Complies with relevant data protection regulations</li>
        </ul>
        <p className="mt-3">
          Images are automatically deleted after processing, and no personal health information is retained.
        </p>
      </section>
    </div>
  );
};

export default AboutPage;
