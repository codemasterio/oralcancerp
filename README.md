# Oral Cancer Detection System

A comprehensive machine learning application for detecting oral cancer from images of the oral cavity.

## Project Overview

This project implements an end-to-end solution for oral cancer detection using deep learning. It includes:

- Data preprocessing and management
- Multiple CNN model architectures (ResNet, EfficientNet, DenseNet, custom CNN)
- Model training, evaluation, and inference
- FastAPI backend for image processing and prediction
- React frontend with intuitive UI for image upload and result visualization

## Project Structure

```
oral-cancer-detection/
├── frontend/                 # React frontend application
├── backend/                  # FastAPI backend server
├── model/                    # ML model training and inference
├── data/                     # Dataset and data processing
├── utils/                    # Utility functions and helpers
├── config/                   # Configuration files
├── tests/                    # Unit and integration tests
├── docs/                     # Documentation
├── requirements/             # Dependency files
└── docker/                   # Docker configuration
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 14+
- pip
- npm or yarn

### Backend Setup

1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`

3. Install dependencies:
   ```
   pip install -r requirements/requirements.txt
   ```

4. Run the backend server:
   ```
   cd backend
   uvicorn main:app --reload
   ```

The API will be available at `http://localhost:8000`.

### Frontend Setup

1. Install dependencies:
   ```
   cd frontend
   npm install
   ```

2. Run the development server:
   ```
   npm start
   ```

The frontend will be available at `http://localhost:3000`.

## Model Training

To train a model:

1. Prepare your dataset in the `data/raw` directory with class subdirectories.
2. Run the data preprocessing script:
   ```
   cd data
   python preprocessing.py
   ```

3. Train a model:
   ```
   cd model/training
   python trainer.py
   ```

## API Endpoints

- `GET /health`: Health check
- `GET /health/status`: Detailed system status
- `GET /health/model-info`: Model information
- `POST /predict`: Upload an image and get a prediction

## Deployment Instructions

### Deploying to Heroku

1. Install the Heroku CLI if you haven't already:
   - Windows: Download from https://devcenter.heroku.com/articles/heroku-cli
   - macOS: `brew install heroku`
   - Linux: Follow instructions at https://devcenter.heroku.com/articles/heroku-cli

2. Login to Heroku:
   ```
   heroku login
   ```

3. Create a new Heroku app:
   ```
   heroku create your-app-name
   ```

4. Push your code to Heroku:
   ```
   git push heroku main
   ```

5. Scale the application:
   ```
   heroku ps:scale web=1
   ```

6. Your application will be available at `https://your-app-name.herokuapp.com`

### Environment Variables

No environment variables are required for basic deployment. The application uses default settings for development.

### Troubleshooting

If you encounter any issues, check the Heroku logs:
```
heroku logs --tail
```

- `GET /health`: Health check
- `GET /health/status`: Detailed system status
- `GET /health/model-info`: Model information
- `POST /predict`: Upload an image and get a prediction
- `GET /predict/visualization/{filename}`: Get prediction visualization
- `POST /predict/batch`: Batch predict multiple images

## Performance Metrics

Our models aim to achieve:
- Accuracy: >90%
- Sensitivity: >95%
- Specificity: >85%
- F1-Score: >90%

## Medical Disclaimer

This tool is designed for screening purposes only and should not replace professional medical advice, diagnosis, or treatment. Always seek the advice of your dentist, physician, or other qualified health provider with any questions regarding a medical condition.

## License

[MIT License](LICENSE)

## Acknowledgements

- TensorFlow and Keras for deep learning frameworks
- FastAPI for the backend API
- React for the frontend UI
- OpenCV for image processing
