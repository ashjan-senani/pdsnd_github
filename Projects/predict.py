# Imports
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import json

def parse_arguments():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Predict flower name from an image using a trained model.")
    parser.add_argument('image_path', type=str, help="Path to the input image.")  # Required: Image file path
    parser.add_argument('model_path', type=str, help="Path to the trained Keras model.")  # Required: Model file path
    parser.add_argument('--top_k', type=int, default=5, help="Number of top predictions to return.")  # Optional: Top K predictions
    parser.add_argument('--category_names', type=str, help="Path to JSON file mapping labels to flower names.")  # Optional: Label map file path
    return parser.parse_args()

def process_image(image):
    """
    Preprocess the input image:
    - Resize to (224, 224)
    - Normalize pixel values to range [0, 1]
    """
    image = tf.convert_to_tensor(image)  # Convert to a TensorFlow tensor
    image = tf.image.resize(image, (224, 224))  # Resize to 224x224
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image.numpy()  # Convert back to a NumPy array

def predict(image_path, model, top_k):
    """
    Predict the top K classes for an image using the trained model.
    Args:
    - image_path: Path to the input image.
    - model: Trained Keras model.
    - top_k: Number of top predictions to return.
    Returns:
    - top_k_probs: Probabilities of the top K classes.
    - top_k_classes: Class indices of the top K predictions.
    """
    # Load and preprocess the image
    image = Image.open(image_path)
    image = np.asarray(image)  # Convert image to a NumPy array
    processed_image = process_image(image)  # Preprocess the image
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(processed_image)
    top_k_probs = np.sort(predictions[0])[-top_k:][::-1]  # Get top K probabilities
    top_k_classes = np.argsort(predictions[0])[-top_k:][::-1]  # Get top K class indices

    return top_k_probs, top_k_classes
def load_class_names(json_path):
    """
    Load label-to-class mappings from a JSON file.
    Args:
    - json_path: Path to the JSON file.
    Returns:
    - class_names: Dictionary mapping labels to class names.
    """
    with open(json_path, 'r') as f:
        return json.load(f)
def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Load the trained model
    model = load_model(args.model_path, custom_objects={'KerasLayer': tf.keras.layers.Layer})

    # Predict the top K classes
    top_k_probs, top_k_classes = predict(args.image_path, model, args.top_k)

    # If category names JSON is provided, map indices to names
    if args.category_names:
        class_names = load_class_names(args.category_names)
        top_k_class_names = [class_names[str(cls)] for cls in top_k_classes]
    else:
        top_k_class_names = top_k_classes

    # Print the results
    print("Top Predictions:")
    for i in range(len(top_k_probs)):
        print(f"{top_k_class_names[i]}: {top_k_probs[i]:.4f}")

# Run the script
if __name__ == '__main__':
    main()
