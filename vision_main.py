# Training utilities
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from vision_transformer import VisionTransformer


def get_dataset(dataset_name='cifar10', batch_size=32, num_workers=2):
    """
    Get MNIST or CIFAR-10 dataset with appropriate transforms
    
    Args:
        dataset_name: 'mnist' or 'cifar10'
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader, test_loader, num_classes, img_size, in_channels
    """
    if dataset_name.lower() == 'mnist':
        # MNIST transforms
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),  # Resize to 32x32 for easier patching
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Download and load MNIST
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform_test
        )
        
        num_classes = 10
        img_size = 32
        in_channels = 1
        
    elif dataset_name.lower() == 'cifar10':
        # CIFAR-10 transforms with data augmentation
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Download and load CIFAR-10
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
        
        num_classes = 10
        img_size = 32
        in_channels = 3

    elif dataset_name.lower() == 'fashion':
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),  # Resize to 32x32 for easier patching
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Download and load FashionMNIST
        train_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=transform_test
        )
        
        num_classes = 10
        img_size = 32
        in_channels = 1
        
    else:
        raise ValueError("Dataset must be 'mnist' or 'cifar10'")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader, num_classes, img_size, in_channels

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, targets) in enumerate(pbar):
        targets = targets.to(device)
        one_hot_targets = F.one_hot(targets, num_classes=num_classes).float()
        data, one_hot_targets = data.to(device), one_hot_targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass - only use classification output for training
        outputs = model(data)
        loss = criterion(outputs, one_hot_targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.3f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(train_loader), 100. * correct / total

def test_epoch(model, test_loader, criterion, device):
    """Test for one epoch"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for data, targets in pbar:
            targets = targets.to(device)
            one_hot_targets = F.one_hot(targets, num_classes=num_classes).float()
            data, one_hot_targets = data.to(device), one_hot_targets.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, one_hot_targets)
            
            # Statistics
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return test_loss / len(test_loader), 100. * correct / total



# Main training and evaluation pipeline
if __name__ == "__main__":
    # Configuration
    DATASET = 'fashion'  # Change to 'mnist' for MNIST dataset
    BATCH_SIZE = 64
    EPOCHS = 150
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    CURVATURES = [0.0, 0.0001, 1.0, 10.0]
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get dataset
    print(f"Loading {DATASET.upper()} dataset...")
    train_loader, test_loader, num_classes, img_size, in_channels = get_dataset(
        DATASET, BATCH_SIZE, num_workers=2
    )
    
    print(f"Dataset: {DATASET.upper()}")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Input channels: {in_channels}")
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    print("======================================\n")

    for c in CURVATURES:
        print(f'Curvature: {c}')
        # Create model optimized for smaller images
        model = VisionTransformer(
            img_size=img_size,      # 32x32 for both datasets
            patch_size=4,           # Smaller patches for 32x32 images
            in_channels=in_channels, # 1 for MNIST, 3 for CIFAR-10
            embed_dim=256,          # Smaller embedding for efficiency
            depth=6,       # Fewer layers for smaller dataset
            num_heads=8,            # Fewer heads
            dropout=0.1,
            num_classes=num_classes,
            mlp_dim = 1024
        ).to(device)
        
        
        # Loss function and optimizer
        criterion = nn.MSELoss(reduction='mean')
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        
        # Training loop
        print(f"\nStarting training for {EPOCHS} epochs...")
        train_losses, train_accs = [], []
        test_losses, test_accs = [], []
        best_acc = 0.0
        
        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            
            # Test
            test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
            
            # Update scheduler
            scheduler.step()
            
            # Save metrics
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_acc': test_acc,
                }, f'best_vit_{DATASET}.pth')
                print(f"New best model saved! Test Acc: {best_acc:.2f}%")
        
        print(f"\nTraining completed!")
        print(f"Best test accuracy: {best_acc:.2f}%")
        
        # Example of using encoder-decoder for other tasks
        print("\nExample: Using encoder-decoder for feature extraction...")
        model.eval()
        with torch.no_grad():
            # Get a sample batch
            sample_images, _ = next(iter(test_loader))
            sample_images = sample_images[:4].to(device)  # Take 4 samples
            
            # Forward pass
            outputs = model(sample_images)
            
            print(f"Sample input shape: {sample_images.shape}")
            print(f"Classification output shape: {outputs.shape}")
            
            # Show attention visualization could be added here
            print("\nModel ready for inference!")
        
        print(f"\nFiles saved:")
        print(f"- best_vit_{DATASET}.pth (best model checkpoint)")

        filename = '/curvatures/'+ DATASET +'_curvature_results_'+ str(c) +'.csv'
        df = pd.DataFrame({
            'test': test_losses
        })

        # Save to CSV
        df.to_csv(os.getcwd() + filename, index=False)
    