"""
Oral Cancer Detection - Model Evaluation Module
This module handles model evaluation, metrics calculation, and visualization.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report
)
import seaborn as sns
import pandas as pd
import logging
import sys
import json
from tensorflow.keras.models import load_model
import itertools

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OralCancerModelEvaluator:
    """
    Class to handle model evaluation for oral cancer detection.
    """
    
    def __init__(self, model_path=None, model=None):
        """
        Initialize the evaluator.
        
        Args:
            model_path (str): Path to the saved model
            model (tf.keras.Model): Model instance (alternative to model_path)
        """
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = self._load_model(model_path)
        else:
            raise ValueError("Either model_path or model must be provided")
        
        # Evaluation results
        self.y_true = None
        self.y_pred = None
        self.y_pred_proba = None
        self.metrics = {}
        
    def _load_model(self, model_path):
        """
        Load a saved model.
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            tf.keras.Model: Loaded model
        """
        logger.info(f"Loading model from {model_path}")
        return load_model(model_path)
    
    def evaluate_generator(self, test_generator):
        """
        Evaluate the model on a test data generator.
        
        Args:
            test_generator: Test data generator
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info("Evaluating model on test data...")
        
        # Get the number of samples in the generator
        num_samples = test_generator.samples
        num_batches = int(np.ceil(num_samples / test_generator.batch_size))
        
        # Reset the generator
        test_generator.reset()
        
        # Get true labels
        y_true = test_generator.classes
        
        # Make predictions
        y_pred_proba = self.model.predict(test_generator, steps=num_batches)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Store results for later use
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_true, y_pred, y_pred_proba)
        self.metrics = metrics
        
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            dict: Evaluation metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Specificity (true negative rate)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        
        # Compile metrics
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'sensitivity': float(recall),  # Same as recall
            'specificity': float(specificity),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            }
        }
        
        logger.info(f"Evaluation metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, "
                   f"Recall/Sensitivity={recall:.4f}, Specificity={specificity:.4f}, "
                   f"F1-Score={f1:.4f}, ROC-AUC={roc_auc:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, save_path=None, class_names=None):
        """
        Plot confusion matrix.
        
        Args:
            save_path (str): Path to save the plot
            class_names (list): Names of the classes
        """
        if self.metrics is None or 'confusion_matrix' not in self.metrics:
            logger.warning("No confusion matrix available to plot.")
            return
        
        if class_names is None:
            class_names = ['Non-Cancer', 'Cancer']
        
        cm = np.array(self.metrics['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Confusion matrix plot saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, save_path=None):
        """
        Plot ROC curve.
        
        Args:
            save_path (str): Path to save the plot
        """
        if self.metrics is None or 'roc_curve' not in self.metrics:
            logger.warning("No ROC curve data available to plot.")
            return
        
        fpr = np.array(self.metrics['roc_curve']['fpr'])
        tpr = np.array(self.metrics['roc_curve']['tpr'])
        roc_auc = self.metrics['roc_auc']
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"ROC curve plot saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, save_path=None):
        """
        Plot precision-recall curve.
        
        Args:
            save_path (str): Path to save the plot
        """
        if self.y_true is None or self.y_pred_proba is None:
            logger.warning("No prediction data available to plot precision-recall curve.")
            return
        
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_pred_proba)
        average_precision = average_precision_score(self.y_true, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall curve: AP={average_precision:.2f}')
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Precision-recall curve plot saved to {save_path}")
        
        plt.show()
    
    def plot_prediction_distribution(self, save_path=None):
        """
        Plot distribution of prediction probabilities.
        
        Args:
            save_path (str): Path to save the plot
        """
        if self.y_true is None or self.y_pred_proba is None:
            logger.warning("No prediction data available to plot distribution.")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Separate predictions by true class
        cancer_probs = self.y_pred_proba[self.y_true == 1].flatten()
        non_cancer_probs = self.y_pred_proba[self.y_true == 0].flatten()
        
        # Plot histograms
        plt.hist(cancer_probs, alpha=0.5, bins=20, range=(0, 1), 
                label='Cancer (True Positive)', color='red')
        plt.hist(non_cancer_probs, alpha=0.5, bins=20, range=(0, 1), 
                label='Non-Cancer (True Negative)', color='green')
        
        plt.xlabel('Prediction Probability')
        plt.ylabel('Count')
        plt.title('Distribution of Prediction Probabilities by True Class')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Prediction distribution plot saved to {save_path}")
        
        plt.show()
    
    def save_metrics(self, save_path):
        """
        Save evaluation metrics to a JSON file.
        
        Args:
            save_path (str): Path to save the metrics
        """
        if self.metrics is None:
            logger.warning("No metrics available to save.")
            return
        
        # Create a copy of metrics for serialization
        metrics_to_save = self.metrics.copy()
        
        # Remove non-serializable objects
        if 'classification_report' in metrics_to_save:
            metrics_to_save['classification_report'] = dict(metrics_to_save['classification_report'])
        
        with open(save_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=4)
        
        logger.info(f"Evaluation metrics saved to {save_path}")
    
    def generate_evaluation_report(self, save_dir):
        """
        Generate a comprehensive evaluation report with all plots and metrics.
        
        Args:
            save_dir (str): Directory to save the report
        """
        if self.metrics is None:
            logger.warning("No metrics available to generate report.")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metrics
        self.save_metrics(os.path.join(save_dir, 'metrics.json'))
        
        # Save plots
        self.plot_confusion_matrix(
            save_path=os.path.join(save_dir, 'confusion_matrix.png')
        )
        self.plot_roc_curve(
            save_path=os.path.join(save_dir, 'roc_curve.png')
        )
        self.plot_precision_recall_curve(
            save_path=os.path.join(save_dir, 'precision_recall_curve.png')
        )
        self.plot_prediction_distribution(
            save_path=os.path.join(save_dir, 'prediction_distribution.png')
        )
        
        # Generate HTML report
        html_report = self._generate_html_report()
        with open(os.path.join(save_dir, 'evaluation_report.html'), 'w') as f:
            f.write(html_report)
        
        logger.info(f"Evaluation report generated in {save_dir}")
    
    def _generate_html_report(self):
        """
        Generate an HTML report of the evaluation results.
        
        Returns:
            str: HTML report
        """
        # Extract metrics
        accuracy = self.metrics['accuracy']
        precision = self.metrics['precision']
        recall = self.metrics['recall']
        specificity = self.metrics['specificity']
        f1 = self.metrics['f1_score']
        roc_auc = self.metrics['roc_auc']
        
        # Create HTML content
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Oral Cancer Detection Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #3498db; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 8px; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metric-card {{ background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .metric-name {{ font-size: 14px; color: #7f8c8d; }}
                .metrics-container {{ display: flex; flex-wrap: wrap; gap: 15px; }}
                .metric-card {{ flex: 1; min-width: 200px; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; margin-top: 10px; }}
            </style>
        </head>
        <body>
            <h1>Oral Cancer Detection Model Evaluation Report</h1>
            
            <h2>Performance Metrics</h2>
            <div class="metrics-container">
                <div class="metric-card">
                    <div class="metric-name">Accuracy</div>
                    <div class="metric-value">{accuracy:.4f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">Precision</div>
                    <div class="metric-value">{precision:.4f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">Recall/Sensitivity</div>
                    <div class="metric-value">{recall:.4f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">Specificity</div>
                    <div class="metric-value">{specificity:.4f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">F1-Score</div>
                    <div class="metric-value">{f1:.4f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">ROC-AUC</div>
                    <div class="metric-value">{roc_auc:.4f}</div>
                </div>
            </div>
            
            <h2>Confusion Matrix</h2>
            <p>The confusion matrix shows the counts of true positives, false positives, true negatives, and false negatives.</p>
            <img src="confusion_matrix.png" alt="Confusion Matrix">
            
            <h2>ROC Curve</h2>
            <p>The Receiver Operating Characteristic (ROC) curve shows the trade-off between sensitivity and specificity.</p>
            <img src="roc_curve.png" alt="ROC Curve">
            
            <h2>Precision-Recall Curve</h2>
            <p>The Precision-Recall curve shows the trade-off between precision and recall for different thresholds.</p>
            <img src="precision_recall_curve.png" alt="Precision-Recall Curve">
            
            <h2>Prediction Distribution</h2>
            <p>This histogram shows the distribution of prediction probabilities for cancer and non-cancer cases.</p>
            <img src="prediction_distribution.png" alt="Prediction Distribution">
            
            <h2>Classification Report</h2>
            <table>
                <tr>
                    <th>Class</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Support</th>
                </tr>
        """
        
        # Add classification report rows
        for class_label, metrics in self.metrics['classification_report'].items():
            if class_label in ['0', '1']:
                class_name = 'Non-Cancer' if class_label == '0' else 'Cancer'
                html += f"""
                <tr>
                    <td>{class_name}</td>
                    <td>{metrics['precision']:.4f}</td>
                    <td>{metrics['recall']:.4f}</td>
                    <td>{metrics['f1-score']:.4f}</td>
                    <td>{metrics['support']}</td>
                </tr>
                """
        
        # Add accuracy row
        html += f"""
                <tr>
                    <td colspan="3"><strong>Accuracy</strong></td>
                    <td>{self.metrics['accuracy']:.4f}</td>
                    <td>{sum(self.metrics['classification_report'][c]['support'] for c in ['0', '1'])}</td>
                </tr>
            </table>
            
            <h2>Medical Interpretation</h2>
            <p>
                <strong>Sensitivity (Recall):</strong> {recall:.4f} - This indicates that the model correctly identifies {recall*100:.1f}% of all actual cancer cases.
                This is a critical metric for medical applications as it represents the model's ability to catch true positives.
            </p>
            <p>
                <strong>Specificity:</strong> {specificity:.4f} - This indicates that the model correctly identifies {specificity*100:.1f}% of all non-cancer cases.
                This represents the model's ability to avoid false positives.
            </p>
            <p>
                <strong>Precision:</strong> {precision:.4f} - When the model predicts cancer, it is correct {precision*100:.1f}% of the time.
                This is important for avoiding unnecessary treatments or anxiety from false positives.
            </p>
            <p>
                <strong>Clinical Implications:</strong> Based on these metrics, this model would be most suitable for 
                {'initial screening' if recall > 0.9 else 'secondary confirmation'} in a clinical workflow.
            </p>
            
            <footer>
                <p><small>Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
            </footer>
        </body>
        </html>
        """
        
        return html

def evaluate_model(model_path, test_data_dir, batch_size=32, image_size=(224, 224), 
                  output_dir=None):
    """
    Convenience function to evaluate a model.
    
    Args:
        model_path (str): Path to the saved model
        test_data_dir (str): Directory containing test data
        batch_size (int): Batch size for evaluation
        image_size (tuple): Image size for the model
        output_dir (str): Directory to save evaluation results
        
    Returns:
        dict: Evaluation metrics
    """
    # Import here to avoid circular imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from data.preprocessing import load_data_generators
    
    # Create data generators
    _, _, test_generator = load_data_generators(
        test_data_dir,  # Placeholder, not used
        test_data_dir,  # Placeholder, not used
        test_data_dir,
        batch_size=batch_size,
        image_size=image_size
    )
    
    # Create evaluator
    evaluator = OralCancerModelEvaluator(model_path=model_path)
    
    # Evaluate the model
    metrics = evaluator.evaluate_generator(test_generator)
    
    # Generate evaluation report if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        evaluator.generate_evaluation_report(output_dir)
    
    return metrics

if __name__ == "__main__":
    # Example usage
    model_path = "../checkpoints/resnet50_20250524-123456_best.h5"
    test_data_dir = "../data/splits/test"
    output_dir = "../evaluation/results"
    
    metrics = evaluate_model(
        model_path=model_path,
        test_data_dir=test_data_dir,
        output_dir=output_dir
    )
