from ultralytics import YOLO
import os

def classify_melanoma(model_path, image_path):
    try:
        # 1. Load your custom trained model
        model = YOLO(model_path)

        # 2. Run classification on the image
        results = model(image_path, verbose=False)

        # 3. Process the results
        result = results[0]
        
        # Get the index of the top prediction
        top_class_index = result.probs.top1
        
        # Get the confidence score (convert tensor to float)
        top_confidence = result.probs.top1conf.item()

        # Get the actual class name ('melanoma' or 'not_melanoma')
        class_name = result.names[top_class_index]

        # 4. Format the text response
        response = f"Diagnosis: {class_name} (Confidence: {top_confidence:.2%})"
        
        return response

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    # --- CONFIGURATION ---
    
    # This path is created automatically after you run train_model.py
    YOUR_MODEL_PATH = 'runs/classify/melanoma_run/weights/best.pt' 

    # REPLACE THIS with the path to an actual image you want to test
    TEST_IMAGE_PATH = 'dataset/val/melanoma/your_test_image.jpg' 

    # --- EXECUTION ---
    
    # Check if model exists before running
    if os.path.exists(YOUR_MODEL_PATH) and os.path.exists(TEST_IMAGE_PATH):
        response_text = classify_melanoma(YOUR_MODEL_PATH, TEST_IMAGE_PATH)
        print(response_text)
    else:
        if not os.path.exists(YOUR_MODEL_PATH):
            print("Error: Model not found. Did you run train_model.py yet?")
        if not os.path.exists(TEST_IMAGE_PATH):
            print(f"Error: Test image not found at {TEST_IMAGE_PATH}")