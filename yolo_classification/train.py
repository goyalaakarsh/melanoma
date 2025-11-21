from ultralytics import YOLO

def train():
    # 1. Load the pretrained YOLO11 classification model
    # 'yolo11s-cls.pt' is the small version (fastest). 
    model = YOLO('yolo11s-cls.pt') 

    # 2. Train the model
    # We specifically name the project 'melanoma_model' so we know where to find it later
    model.train(
        data='dataset', # Name of your folder containing 'train' and 'val'
        epochs=20,               # How many times to go through the images
        imgsz=224,               # Standard image size for classification
        project='runs/classify', # Save results here
        name='melanoma_run'      # Name of this specific training run
    )
    
    print("Training Finished!")
    print("Your new model is saved at: runs/classify/melanoma_run/weights/best.pt")

if __name__ == '__main__':
    train()