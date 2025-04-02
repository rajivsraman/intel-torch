import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import csv
from trainmodel import create_model  # reusing the model architecture from training

def load_and_prepare_image(image_path):
    # Same preprocessing as in training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def predict_image(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create and load the trained model
    num_classes = 6
    model = create_model(num_classes)
    model.load_state_dict(torch.load('best_model.pth'))
    model = model.to(device)
    model.eval()
    
    # Directory containing test images
    pred_dir = "data/seg_pred/seg_pred"
    
    # Get class names from training data structure
    class_names = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
    
    # Create a list to store results
    results = []
    
    print("The model is making predictions...")

    # Process all images in the prediction directory
    for image_name in os.listdir(pred_dir):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(pred_dir, image_name)
            
            # Prepare image
            image_tensor = load_and_prepare_image(image_path)
            
            # Get prediction
            pred_idx = predict_image(model, image_tensor, device)
            predicted_class = class_names[pred_idx]
            
            # Store result
            results.append([image_name, predicted_class])
    
    # Save results to CSV
    with open('predictions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'Predicted Class'])  # Write header
        writer.writerows(results)
    
    print("Predictions have been saved to 'predictions.csv'")

if __name__ == '__main__':
    main()
