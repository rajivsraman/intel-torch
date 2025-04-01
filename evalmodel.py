import torch
import torchvision
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from trainmodel import create_model
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

def evaluate_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create and load the trained model
    num_classes = 6
    model = create_model(num_classes)
    model.load_state_dict(torch.load('best_model.pth'))
    model.to(device)
    model.eval()
    
    # Validation transforms
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load validation dataset
    val_dataset = datasets.ImageFolder(root="data/seg_test/seg_test", transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Lists to store predictions and true labels
    all_preds = []
    all_labels = []
    
    # Get predictions
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Get class names
    class_names = val_dataset.classes
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Calculate additional metrics
    accuracy = np.sum(all_preds == all_labels) / len(all_labels)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Calculate per-class accuracy
    print("\nPer-class Accuracy:")
    for i, class_name in enumerate(class_names):
        mask = all_labels == i
        class_acc = np.sum(all_preds[mask] == all_labels[mask]) / np.sum(mask)
        print(f"{class_name}: {class_acc:.4f}")

if __name__ == '__main__':
    evaluate_model()