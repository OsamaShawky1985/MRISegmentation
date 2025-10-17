import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

import yaml
import numpy as np
from ExtractionPipeline import ExtractionPipeline
from models.MLPDetector import MLPDetector
from ExtractionPipeline import ExtractionPipeline
import tensorflow as tf

import matplotlib.pyplot as plt


def plot_accuracy_epochs(history, save_path='data/accuracy_epochs.png'):
    """
    Plot training and validation accuracy over epochs
    
    Args:
        history: Keras history object containing training history
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Plot training & validation accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    
    # Customize the plot
    plt.title('Model Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()


def main():
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Stage 1: Feature Extraction
    print("Extracting features...")
    # Create pipeline (now includes feature extractor)
    pipeline = ExtractionPipeline(config)
    
    # Process all images
    
    # pipeline.process_directory(
    #     input_dir='data/raw_data',
    #     output_dir='data/features'
    # )
    print("Feature extraction completed. Features saved to 'data/features/features.npy' and labels to 'data/features/labels.npy'.")
    #print("DEBUG: Feature extraction completed. Features saved to 'data/features/features.npy' and labels to 'data/features/labels.npy'.")
    #return
    # Load extracted features
    features = np.load('data/features/features.npy')
    labels = np.load('data/features/labels.npy')
    labels = (labels == 'BrainTumor').astype(int)
    print("DEBUG: labels type:", type(labels), "shape:", getattr(features, 'shape', None))
    #return
    print("Distinct classes:", set(labels))
    print("Number of distinct classes:", len(set(labels)))
    #return
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    print(f"Number of training images: {len(X_train)}")
    print(f"Number of testing images: {len(X_test)}")
    # Stage 2: MLP Training
    print("Training MLP detector...")
    detector = MLPDetector(config)
    model = detector.build_model(input_shape=features.shape[1])
    
    history = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=100,
        validation_data=(X_test, y_test),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10),
            tf.keras.callbacks.ModelCheckpoint(
                'models/mlp_detector.h5',
                save_best_only=True
            )
        ]
    )
    plot_accuracy_epochs(history)
    # Evaluate model on test set
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,recall_score, f1_score
    import matplotlib.pyplot as plt
    import seaborn as sns

    # For binary classification
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)  # For sigmoid output

# For multi-class classification (softmax output)
# y_pred_prob = model.predict(X_test)
# y_pred = y_pred_prob.argmax(axis=1)
# y_true = y_test.argmax(axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='binary')  # use 'macro' or 'weighted' for multi-class
    f1 = f1_score(y_test, y_pred, average='binary')          # use 'macro' or 'weighted' for multi-class

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    y_pred = model.predict(X_test)
    # If model outputs probabilities, convert to class labels
    if y_pred.shape[-1] > 1:
        y_pred = y_pred.argmax(axis=1)
    else:
        y_pred = (y_pred > 0.5).astype(int).flatten()
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    # Draw confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy','BrainTumor'], yticklabels=['Healthy','BrainTumor'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Save features for future use
    np.save('models/feature_scaler.npy', features.mean(axis=0))

if __name__ == '__main__':
    main()

