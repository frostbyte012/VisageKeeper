#!/usr/bin/env python3
"""
VisageKeeper / FRPure - Individual Image Exporter
Saves every pipeline stage and intermediate denoiser layer as individual image files
in a structured output folder.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Pipeline parameters
EPSILON = 0.1                    
SIGMA = 0.5                      
N_SAMPLES = 4                    
IMG_SIZE = 112                   
OUTPUT_DIR = "visagekeeper_exported_images"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# SIMPLE DNCNN DENOISER
# ============================================================================

class SimpleDnCNN(nn.Module):
    """Simplified DnCNN denoiser - residual learning"""
    def __init__(self, in_channels=3, out_channels=3, num_layers=4):
        super(SimpleDnCNN, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        self.relu_layers = nn.ModuleList()
        
        self.conv_layers.append(nn.Conv2d(in_channels, 32, kernel_size=3, padding=1))
        self.relu_layers.append(nn.ReLU(inplace=False))
        
        for _ in range(num_layers - 2):
            self.conv_layers.append(nn.Conv2d(32, 32, kernel_size=3, padding=1))
            self.relu_layers.append(nn.ReLU(inplace=False))
        
        self.conv_layers.append(nn.Conv2d(32, out_channels, kernel_size=3, padding=1))
    
    def forward(self, x, return_intermediates=False):
        intermediates = []
        current = x
        
        # Capture intermediate Conv+ReLU layers (32 channels)
        for i, (conv, relu) in enumerate(zip(self.conv_layers[:-1], self.relu_layers)):
            current = conv(current)
            current = relu(current)
            intermediates.append(current.detach())
        
        # Capture Layer 4: Conv (Predicted Noise, 3 channels)
        predicted_noise = self.conv_layers[-1](current)
        intermediates.append(predicted_noise.detach())
        
        # Capture Layer 5: Subtraction (Input - Noise)
        subtracted = x - predicted_noise
        intermediates.append(subtracted.detach())
        
        # Capture Layer 6: Clamp
        clamped_output = torch.clamp(subtracted, 0, 1)
        intermediates.append(clamped_output.detach())
        
        if return_intermediates:
            return clamped_output, intermediates
        return clamped_output

# ============================================================================
# SIMPLE FR BACKBONE (Mock)
# ============================================================================

class SimpleFRBackbone(nn.Module):
    def __init__(self, embedding_dim=512):
        super(SimpleFRBackbone, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.head = nn.Linear(128, embedding_dim)
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return F.normalize(x, p=2, dim=1)

# ============================================================================
# ATTACK & DEFENSE FUNCTIONS
# ============================================================================

def fgsm_attack(image, model, target_embedding, epsilon):
    image.requires_grad = True
    embedding = model(image)
    loss = -F.cosine_similarity(embedding.unsqueeze(0), target_embedding.unsqueeze(0)).mean()
    model.zero_grad()
    loss.backward()
    perturbation = epsilon * image.grad.sign()
    return torch.clamp(image.detach() + perturbation, 0, 1)

def randomized_smoothing(image, sigma, n_samples):
    noisy_images = []
    for _ in range(n_samples):
        noise = torch.randn_like(image) * sigma
        noisy = torch.clamp(image + noise, 0, 1)
        noisy_images.append(noisy)
    return torch.cat(noisy_images, dim=0)

# ============================================================================
# PROCESSING HELPERS
# ============================================================================

def tensor_to_pil(tensor):
    """Converts standard images OR 32-channel feature maps to a savable PIL image."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    tensor = tensor.detach().cpu()
    
    # If tensor has 32 channels (intermediate denoiser layers), average them to make a grayscale heatmap
    if tensor.shape[0] > 3:
        tensor = tensor.mean(dim=0, keepdim=True)
        
    # Normalize tensor to 0-255 range for clean contrast visibility
    t_min, t_max = tensor.min(), tensor.max()
    if t_max > t_min:
        tensor = (tensor - t_min) / (t_max - t_min)
        
    tensor = (tensor * 255).clamp(0, 255).numpy().astype(np.uint8)
    
    if tensor.shape[0] == 3:
        tensor = np.transpose(tensor, (1, 2, 0))
        return Image.fromarray(tensor)
    else:
        return Image.fromarray(tensor.squeeze(0), mode='L')

def save_text_as_image(title, body_text, filename):
    """Generates a clean metadata image for pipeline steps that output numerical data"""
    img = Image.new('RGB', (400, 400), color='#f8f9fa')
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    
    # Draw simple fallback borders and layout text blocks
    d.rectangle([(10, 10), (390, 390)], outline="gray", width=2)
    d.text((20, 30), title, fill="black")
    d.text((20, 100), body_text, fill="blue")
    
    img.save(os.path.join(OUTPUT_DIR, filename))

# ============================================================================
# MAIN EXPORT PIPELINE
# ============================================================================

def export_pipeline_images():
    print(f"Executing pipeline and exporting individual components to target folder: '{OUTPUT_DIR}/'...")
    
    # ------------------------------------------------------------------------
    # 0. Load Input Face Image
    # ------------------------------------------------------------------------
    image_path = "triz.jpg" # <--- Change this to your test file path if needed
    
    if os.path.exists(image_path):
        print(f"Found user image: '{image_path}'. Loading...")
        raw_pil = Image.open(image_path).convert('RGB')
    else:
        print(f"Image '{image_path}' not found. Falling back to generating a clear pattern asset...")
        # Fallback to a structured pattern asset rather than plain white noise for clearer visual tracing
        fallback_arr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        fallback_arr[30:80, 30:80, 0] = 255 # Red square patch
        fallback_arr[10:50, 60:100, 1] = 255 # Green square patch
        raw_pil = Image.fromarray(fallback_arr)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    clean_image = transform(raw_pil).unsqueeze(0).to(DEVICE)
    
    # Initialize networks
    denoiser = SimpleDnCNN().to(DEVICE).eval()
    fr_backbone = SimpleFRBackbone().to(DEVICE).eval()
    
    with torch.no_grad():
        gallery_embedding = fr_backbone(clean_image)
    
    # ------------------------------------------------------------------------
    # 1. Process Core Pipeline Data
    # ------------------------------------------------------------------------
    # Step 1: Clean
    tensor_to_pil(clean_image).save(os.path.join(OUTPUT_DIR, "step_1_clean_probe.png"))
    
    # Step 2: Attack
    adv_image = fgsm_attack(clean_image.clone(), fr_backbone, gallery_embedding, EPSILON)
    tensor_to_pil(adv_image).save(os.path.join(OUTPUT_DIR, "step_2_adversarial_attack.png"))
    
    # Step 3: Randomized Smoothing
    noisy_images = randomized_smoothing(adv_image, SIGMA, N_SAMPLES)
    noisy_single = noisy_images[0:1] 
    tensor_to_pil(noisy_single).save(os.path.join(OUTPUT_DIR, "step_3_randomized_smoothing_sample.png"))
    
    # Step 4: Denoiser Rectification
    with torch.no_grad():
        denoised_single, intermediates = denoiser(noisy_single, return_intermediates=True)
        denoised_all = denoiser(noisy_images)
    tensor_to_pil(denoised_single).save(os.path.join(OUTPUT_DIR, "step_4_denoiser_rectified_sample.png"))
    
    # Step 5: Feature Extraction (Save descriptive metadata asset)
    with torch.no_grad():
        embeddings = fr_backbone(denoised_all)
    save_text_as_image("Step 5: FR Backbone Embeddings", 
                       f"Processed {N_SAMPLES} smooth iterations.\nGenerated hypersphere deep vectors.\nMatrix Shape: {list(embeddings.shape)}", 
                       "step_5_fr_backbone_embeddings.png")
    
    # Step 6 & 7: Aggregation and Decision
    similarities = F.cosine_similarity(embeddings, gallery_embedding, dim=1)
    votes = (similarities > 0.5).float()
    n_votes = int(votes.sum().item())
    decision = "VERIFIED" if n_votes > N_SAMPLES / 2 else "REJECTED"
    
    voting_details = f"Similarity scores:\n{similarities.cpu().numpy()}\n\nTotal matching votes: {n_votes}/{N_SAMPLES}\nFinal System Assert: {decision}"
    save_text_as_image("Step 6 & 7: Evaluation Engine", voting_details, "step_6_7_voting_and_decision.png")
    
    # Step 8: Enrolled Target Gallery Reference
    tensor_to_pil(clean_image).save(os.path.join(OUTPUT_DIR, "step_8_enrolled_gallery_b.png"))

    # ------------------------------------------------------------------------
    # 2. Process Internal Denoiser Architecture Sequence
    # ------------------------------------------------------------------------
    denoiser_names = [
        "denoiser_layer_0_noisy_input.png",
        "denoiser_layer_1_conv3x3_relu.png",
        "denoiser_layer_2_conv3x3_relu.png",
        "denoiser_layer_3_conv3x3_relu.png",
        "denoiser_layer_4_predicted_noise_map.png",
        "denoiser_layer_5_residual_subtraction.png",
        "denoiser_layer_6_final_clamped_output.png"
    ]
    
    # Capture the full raw noisy slice as starting layer item
    tensor_to_pil(noisy_single).save(os.path.join(OUTPUT_DIR, denoiser_names[0]))
    
    # Iterate through extracted target tensors
    for idx, feature_map in enumerate(intermediates):
        filename = denoiser_names[idx + 1]
        tensor_to_pil(feature_map).save(os.path.join(OUTPUT_DIR, filename))
        
    print(f"\n All files written successfully! Check your directory contents at: ./{OUTPUT_DIR}/")

if __name__ == '__main__':
    export_pipeline_images()