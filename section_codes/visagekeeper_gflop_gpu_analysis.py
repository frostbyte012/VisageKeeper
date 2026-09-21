#!/usr/bin/env python3
"""
VisageKeeper Algorithm - GFLOP Calculation & GPU Runtime Analysis
Calculates computational cost and measures actual runtime on GPU

Computes:
1. GFLOP per component (denoiser, FR model, voting)
2. Total GFLOP for full pipeline
3. Actual GPU runtime measurements
4. GPU memory usage
5. Performance metrics (GFLOP/sec, throughput)
6. Hardware analysis report
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from torchvision.models import resnet50

# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*80}")
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"{'='*80}\n")

# Algorithm parameters
IMG_SIZE = 112
EMBEDDING_DIM = 512
DENOISER_CHANNELS = 32
DENOISER_LAYERS = 4
N_SAMPLES = 1000  # Number of noise-augmented samples
BATCH_SIZE = 1    # Process one sample at a time (realistic scenario)
NUM_ITERATIONS = 100  # Number of runtime measurements

# ============================================================================
# NEURAL NETWORK MODELS
# ============================================================================

class DnCNNDenoiser(nn.Module):
    """DnCNN denoiser for noise/adversarial removal"""
    def __init__(self, in_channels=3, out_channels=3, num_layers=4, channels=32):
        super(DnCNNDenoiser, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        self.relu_layers = nn.ModuleList()
        
        # First layer: 3 -> channels
        self.conv_layers.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1))
        self.relu_layers.append(nn.ReLU(inplace=True))
        
        # Middle layers: channels -> channels
        for _ in range(num_layers - 2):
            self.conv_layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
            self.relu_layers.append(nn.ReLU(inplace=True))
        
        # Last layer: channels -> out_channels
        self.conv_layers.append(nn.Conv2d(channels, out_channels, kernel_size=3, padding=1))
    
    def forward(self, x):
        """Forward pass"""
        current = x
        for conv, relu in zip(self.conv_layers[:-1], self.relu_layers):
            current = conv(current)
            current = relu(current)
        
        residual = self.conv_layers[-1](current)
        output = torch.clamp(x - residual, 0, 1)
        return output


class SimpleFRBackbone(nn.Module):
    """Simple FR backbone for testing (can replace with ArcFace)"""
    def __init__(self, embedding_dim=512):
        super(SimpleFRBackbone, self).__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.head = nn.Linear(256, embedding_dim)
    
    def forward(self, x):
        """Extract embedding"""
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class ArcFaceBackbone(nn.Module):
    """ResNet50-based FR model (similar to ArcFace)"""
    def __init__(self, embedding_dim=512):
        super(ArcFaceBackbone, self).__init__()
        
        self.backbone = resnet50(pretrained=False)
        self.backbone.fc = nn.Linear(2048, embedding_dim)
    
    def forward(self, x):
        """Extract embedding"""
        x = self.backbone(x)
        x = F.normalize(x, p=2, dim=1)
        return x

# ============================================================================
# GFLOP CALCULATION FUNCTIONS
# ============================================================================

def count_conv_flops(kernel_size: int, in_channels: int, out_channels: int, 
                     input_height: int, input_width: int) -> float:
    """
    Calculate FLOPs for a single convolution layer
    
    FLOP = 2 × kernel_h × kernel_w × in_channels × out_channels × output_h × output_w
    (2 because multiply-accumulate = 2 operations)
    """
    output_height = input_height  # With padding
    output_width = input_width
    
    flops = 2 * kernel_size * kernel_size * in_channels * out_channels * output_height * output_width
    return flops

def count_linear_flops(in_features: int, out_features: int) -> float:
    """
    Calculate FLOPs for fully connected layer
    FLOP = 2 × in_features × out_features
    """
    return 2 * in_features * out_features

def calculate_denoiser_gflops(img_size: int = 112, channels: int = 32, 
                              num_layers: int = 4) -> Dict[str, float]:
    """
    Calculate GFLOP for DnCNN denoiser
    
    Architecture:
    - Conv3x3 (3 → 32)
    - Conv3x3 (32 → 32) × (num_layers - 2)
    - Conv3x3 (32 → 3)
    """
    
    flops_dict = {}
    total_flops = 0
    kernel_size = 3
    
    # Layer 1: Conv3x3 (3 → 32)
    flops = count_conv_flops(kernel_size, 3, channels, img_size, img_size)
    flops_dict['layer_1_conv'] = flops
    total_flops += flops
    
    # Middle layers: Conv3x3 (32 → 32)
    for i in range(num_layers - 2):
        flops = count_conv_flops(kernel_size, channels, channels, img_size, img_size)
        flops_dict[f'layer_{i+2}_conv'] = flops
        total_flops += flops
    
    # Last layer: Conv3x3 (32 → 3)
    flops = count_conv_flops(kernel_size, channels, 3, img_size, img_size)
    flops_dict[f'layer_{num_layers}_conv'] = flops
    total_flops += flops
    
    # Convert to GFLOP (billions)
    gflops = total_flops / 1e9
    flops_dict['total_gflops'] = gflops
    
    return flops_dict

def calculate_fr_backbone_gflops(img_size: int = 112, 
                                 embedding_dim: int = 512,
                                 model_type: str = 'simple') -> Dict[str, float]:
    """
    Calculate GFLOP for FR backbone
    """
    
    flops_dict = {}
    total_flops = 0
    
    if model_type == 'simple':
        # Conv 3→64 (7×7, stride=2)
        flops = count_conv_flops(7, 3, 64, img_size//2, img_size//2)
        flops_dict['conv1'] = flops
        total_flops += flops
        
        # MaxPool 3×3
        # MaxPool doesn't count as FLOPs in many frameworks
        
        # Conv 64→128 (3×3)
        flops = count_conv_flops(3, 64, 128, img_size//4, img_size//4)
        flops_dict['conv2'] = flops
        total_flops += flops
        
        # Conv 128→256 (3×3)
        flops = count_conv_flops(3, 128, 256, img_size//4, img_size//4)
        flops_dict['conv3'] = flops
        total_flops += flops
        
        # AdaptiveAvgPool → 1×1
        
        # FC 256 → embedding_dim
        flops = count_linear_flops(256, embedding_dim)
        flops_dict['fc'] = flops
        total_flops += flops
    
    elif model_type == 'arcface':
        # ResNet50 is much larger
        # Approximate: ResNet50 ≈ 4.1 GFLOP per image
        total_flops = 4.1e9
        flops_dict['resnet50_approx'] = total_flops
    
    gflops = total_flops / 1e9
    flops_dict['total_gflops'] = gflops
    
    return flops_dict

def calculate_cosine_similarity_gflops(embedding_dim: int = 512) -> Dict[str, float]:
    """
    Calculate GFLOP for cosine similarity computation
    
    Formula: sim = embedding1 · embedding2
    Operations: embedding_dim multiply + (embedding_dim-1) add
    """
    
    # Dot product: embedding_dim multiplies + (embedding_dim-1) adds
    flops = 2 * embedding_dim - 1
    gflops = flops / 1e9
    
    return {
        'dot_product_flops': flops,
        'total_gflops': gflops
    }

def calculate_voting_gflops(n_samples: int = 1000, 
                            embedding_dim: int = 512) -> Dict[str, float]:
    """
    Calculate GFLOP for voting/aggregation
    
    Operations:
    - n_samples cosine similarities: n_samples × (2×embedding_dim - 1)
    - Clopper-Pearson computation: negligible (mostly comparisons)
    - Inverse CDF: negligible (lookup table)
    """
    
    flops_dict = {}
    total_flops = 0
    
    # Cosine similarity for n samples
    sim_flops = n_samples * (2 * embedding_dim - 1)
    flops_dict['n_cosine_sims'] = sim_flops
    total_flops += sim_flops
    
    # Voting/counting: O(n_samples)
    # Clopper-Pearson and inverse CDF: negligible
    
    gflops = total_flops / 1e9
    flops_dict['total_gflops'] = gflops
    
    return flops_dict

def calculate_total_pipeline_gflops(n_samples: int = 1000,
                                   img_size: int = 112,
                                   embedding_dim: int = 512,
                                   denoiser_channels: int = 32,
                                   denoiser_layers: int = 4,
                                   model_type: str = 'simple') -> Dict[str, float]:
    """
    Calculate total GFLOP for complete VisageKeeper pipeline
    
    Pipeline:
    1. Denoiser × n_samples: n × denoiser_gflop
    2. FR backbone × (n+1) samples: (n+1) × fr_gflop
    3. Cosine similarity × (n+1) samples: (n+1) × sim_gflop
    4. Voting/certification: voting_gflop
    """
    
    print("\n" + "="*80)
    print("GFLOP CALCULATION - DETAILED BREAKDOWN")
    print("="*80 + "\n")
    
    # Component 1: Gallery embedding extraction (single pass)
    print("[1] Gallery Embedding Extraction (ONCE)")
    print("-" * 80)
    fr_gflops = calculate_fr_backbone_gflops(img_size, embedding_dim, model_type)
    gallery_gflops = fr_gflops['total_gflops']
    print(f"  FR Backbone GFLOP: {gallery_gflops:.4f}")
    print(f"  Operation: Extract 1 clean gallery embedding")
    print()
    
    # Component 2: Denoiser (n_samples times)
    print(f"[2] Denoiser Processing ({n_samples} times)")
    print("-" * 80)
    denoiser_gflops = calculate_denoiser_gflops(img_size, denoiser_channels, denoiser_layers)
    denoiser_per_sample = denoiser_gflops['total_gflops']
    denoiser_total = denoiser_per_sample * n_samples
    print(f"  Per sample GFLOP: {denoiser_per_sample:.4f}")
    print(f"  Total ({n_samples} samples): {denoiser_total:.2f}")
    print()
    
    # Component 3: FR backbone for noisy samples (n_samples times)
    print(f"[3] FR Backbone for Noisy Samples ({n_samples} times)")
    print("-" * 80)
    fr_per_sample = fr_gflops['total_gflops']
    fr_total = fr_per_sample * n_samples
    print(f"  Per sample GFLOP: {fr_per_sample:.4f}")
    print(f"  Total ({n_samples} samples): {fr_total:.2f}")
    print()
    
    # Component 4: Cosine similarity (n_samples times)
    print(f"[4] Cosine Similarity Computation ({n_samples} times)")
    print("-" * 80)
    sim_gflops = calculate_cosine_similarity_gflops(embedding_dim)
    sim_per_sample = sim_gflops['total_gflops']
    sim_total = sim_per_sample * n_samples
    print(f"  Per sample GFLOP: {sim_per_sample:.6f}")
    print(f"  Total ({n_samples} samples): {sim_total:.6f}")
    print()
    
    # Component 5: Voting and certification
    print(f"[5] Voting & Certification")
    print("-" * 80)
    voting_gflops = calculate_voting_gflops(n_samples, embedding_dim)
    voting_total = voting_gflops['total_gflops']
    print(f"  Total GFLOP: {voting_total:.6f}")
    print()
    
    # TOTAL
    total_gflops = (gallery_gflops + denoiser_total + fr_total + 
                   sim_total + voting_total)
    
    print("="*80)
    print("TOTAL PIPELINE GFLOP SUMMARY")
    print("="*80)
    print(f"Gallery extraction:           {gallery_gflops:.4f} GFLOP")
    print(f"Denoiser ({n_samples}×):        {denoiser_total:.2f} GFLOP")
    print(f"FR backbone ({n_samples}×):    {fr_total:.2f} GFLOP")
    print(f"Cosine similarity ({n_samples}×): {sim_total:.6f} GFLOP")
    print(f"Voting/Certification:         {voting_total:.6f} GFLOP")
    print("-" * 80)
    print(f"TOTAL:                        {total_gflops:.2f} GFLOP")
    print("="*80)
    
    return {
        'gallery_gflops': gallery_gflops,
        'denoiser_gflops': denoiser_total,
        'fr_backbone_gflops': fr_total,
        'cosine_similarity_gflops': sim_total,
        'voting_gflops': voting_total,
        'total_gflops': total_gflops,
        'n_samples': n_samples
    }

# ============================================================================
# GPU RUNTIME MEASUREMENT FUNCTIONS
# ============================================================================

def measure_denoiser_runtime(n_iterations: int = 100, 
                            img_size: int = 112,
                            batch_size: int = 1) -> Dict[str, float]:
    """
    Measure actual GPU runtime for denoiser
    """
    
    print("\n" + "="*80)
    print("GPU RUNTIME MEASUREMENT - DENOISER")
    print("="*80 + "\n")
    
    denoiser = DnCNNDenoiser(in_channels=3, out_channels=3, 
                            num_layers=DENOISER_LAYERS).to(DEVICE)
    denoiser.eval()
    
    # Dummy input
    dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(DEVICE)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = denoiser(dummy_input)
    
    # Measurement
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(n_iterations):
            _ = denoiser(dummy_input)
    
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / n_iterations
    
    print(f"Denoiser Runtime Analysis ({n_iterations} iterations):")
    print(f"  Total time: {total_time:.4f} seconds")
    print(f"  Average per image: {avg_time*1000:.2f} ms")
    print(f"  Throughput: {1/avg_time:.2f} images/second")
    print()
    
    return {
        'total_time_sec': total_time,
        'avg_time_per_image_ms': avg_time * 1000,
        'throughput_images_per_sec': 1 / avg_time
    }

def measure_fr_backbone_runtime(n_iterations: int = 100,
                               img_size: int = 112,
                               batch_size: int = 1,
                               model_type: str = 'simple') -> Dict[str, float]:
    """
    Measure actual GPU runtime for FR backbone
    """
    
    print("="*80)
    print(f"GPU RUNTIME MEASUREMENT - FR BACKBONE ({model_type.upper()})")
    print("="*80 + "\n")
    
    if model_type == 'simple':
        fr_model = SimpleFRBackbone(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    else:
        fr_model = ArcFaceBackbone(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    
    fr_model.eval()
    
    # Dummy input
    dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(DEVICE)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = fr_model(dummy_input)
    
    # Measurement
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(n_iterations):
            _ = fr_model(dummy_input)
    
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / n_iterations
    
    print(f"FR Backbone Runtime Analysis ({n_iterations} iterations):")
    print(f"  Total time: {total_time:.4f} seconds")
    print(f"  Average per image: {avg_time*1000:.2f} ms")
    print(f"  Throughput: {1/avg_time:.2f} images/second")
    print()
    
    return {
        'total_time_sec': total_time,
        'avg_time_per_image_ms': avg_time * 1000,
        'throughput_images_per_sec': 1 / avg_time
    }

def measure_cosine_similarity_runtime(n_iterations: int = 1000,
                                     embedding_dim: int = 512) -> Dict[str, float]:
    """
    Measure GPU runtime for cosine similarity
    """
    
    print("="*80)
    print("GPU RUNTIME MEASUREMENT - COSINE SIMILARITY")
    print("="*80 + "\n")
    
    # Dummy embeddings
    e1 = torch.randn(1, embedding_dim).to(DEVICE)
    e2 = torch.randn(1, embedding_dim).to(DEVICE)
    
    # Warmup
    with torch.no_grad():
        for _ in range(100):
            _ = F.cosine_similarity(e1, e2)
    
    # Measurement
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(n_iterations):
            _ = F.cosine_similarity(e1, e2)
    
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / n_iterations
    
    print(f"Cosine Similarity Runtime Analysis ({n_iterations} iterations):")
    print(f"  Total time: {total_time:.4f} seconds")
    print(f"  Average per pair: {avg_time*1e6:.2f} μs (microseconds)")
    print(f"  Operations per second: {1/avg_time:.2e}")
    print()
    
    return {
        'total_time_sec': total_time,
        'avg_time_per_pair_us': avg_time * 1e6,
        'ops_per_sec': 1 / avg_time
    }

def measure_full_pipeline_runtime(n_samples: int = 1000,
                                 n_test_iterations: int = 10,
                                 img_size: int = 112) -> Dict[str, float]:
    """
    Measure full VisageKeeper pipeline runtime
    """
    
    print("="*80)
    print(f"GPU RUNTIME MEASUREMENT - FULL PIPELINE ({n_test_iterations} runs, {n_samples} samples each)")
    print("="*80 + "\n")
    
    # Initialize models
    denoiser = DnCNNDenoiser().to(DEVICE)
    fr_model = SimpleFRBackbone(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    denoiser.eval()
    fr_model.eval()
    
    # Dummy data
    gallery_image = torch.randn(1, 3, img_size, img_size).to(DEVICE)
    test_image = torch.randn(1, 3, img_size, img_size).to(DEVICE)
    
    # Warmup
    with torch.no_grad():
        _ = fr_model(gallery_image)
        for _ in range(5):
            noisy = test_image + torch.randn_like(test_image) * 0.5
            denoised = denoiser(noisy)
            _ = fr_model(denoised)
            _ = F.cosine_similarity(fr_model(denoised), fr_model(gallery_image))
    
    # Measurement
    times = []
    
    for run in range(n_test_iterations):
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        with torch.no_grad():
            # Stage 1: Gallery embedding
            e_gallery = fr_model(gallery_image)
            
            # Stage 2: Certification loop
            scores = []
            for _ in range(n_samples):
                # Add noise
                noise = torch.randn_like(test_image) * 0.5
                noisy = torch.clamp(test_image + noise, 0, 1)
                
                # Denoise
                denoised = denoiser(noisy)
                
                # Extract embedding
                e_probe = fr_model(denoised)
                
                # Compute similarity
                sim = F.cosine_similarity(e_probe, e_gallery)
                scores.append(sim)
        
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        run_time = end_time - start_time
        times.append(run_time)
        
        print(f"  Run {run+1}: {run_time:.2f} seconds")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print()
    print(f"Full Pipeline Runtime Analysis:")
    print(f"  Average: {avg_time:.2f} ± {std_time:.2f} seconds ({n_samples} samples)")
    print(f"  Min: {min_time:.2f} seconds")
    print(f"  Max: {max_time:.2f} seconds")
    print(f"  Per sample: {(avg_time/n_samples)*1000:.2f} ms")
    print()
    
    return {
        'avg_time_sec': avg_time,
        'std_time_sec': std_time,
        'min_time_sec': min_time,
        'max_time_sec': max_time,
        'per_sample_ms': (avg_time / n_samples) * 1000
    }

# ============================================================================
# GPU MEMORY ANALYSIS
# ============================================================================

def analyze_gpu_memory(n_samples: int = 1000,
                      img_size: int = 112) -> Dict[str, float]:
    """
    Analyze GPU memory usage
    """
    
    print("="*80)
    print("GPU MEMORY ANALYSIS")
    print("="*80 + "\n")
    
    if DEVICE.type != 'cuda':
        print("GPU memory analysis requires CUDA")
        return {}
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Initialize models
    denoiser = DnCNNDenoiser().to(DEVICE)
    fr_model = SimpleFRBackbone(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    
    # Get model sizes
    denoiser_params = sum(p.numel() for p in denoiser.parameters())
    fr_params = sum(p.numel() for p in fr_model.parameters())
    
    denoiser_mb = (denoiser_params * 4) / (1024**2)  # 4 bytes per float32
    fr_mb = (fr_params * 4) / (1024**2)
    
    print(f"Model Sizes:")
    print(f"  Denoiser: {denoiser_params:,} parameters ({denoiser_mb:.2f} MB)")
    print(f"  FR Model: {fr_params:,} parameters ({fr_mb:.2f} MB)")
    print()
    
    # Measure peak memory during inference
    dummy_input = torch.randn(1, 3, img_size, img_size).to(DEVICE)
    
    with torch.no_grad():
        for _ in range(n_samples):
            _ = denoiser(dummy_input)
            _ = fr_model(dummy_input)
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # Convert to GB
    
    print(f"Peak GPU Memory Usage (during {n_samples} iterations):")
    print(f"  {peak_memory:.2f} GB")
    print()
    
    return {
        'denoiser_params': denoiser_params,
        'fr_params': fr_params,
        'denoiser_mb': denoiser_mb,
        'fr_mb': fr_mb,
        'peak_memory_gb': peak_memory
    }

# ============================================================================
# PERFORMANCE METRICS COMPUTATION
# ============================================================================

def compute_performance_metrics(gflops_data: Dict,
                               runtime_data: Dict) -> Dict:
    """
    Compute performance metrics: GFLOP/s, efficiency, etc.
    """
    
    print("="*80)
    print("PERFORMANCE METRICS")
    print("="*80 + "\n")
    
    total_gflops = gflops_data['total_gflops']
    total_time = runtime_data['avg_time_sec']
    
    gflops_per_sec = total_gflops / total_time
    n_samples = gflops_data['n_samples']
    samples_per_sec = n_samples / total_time
    
    print(f"Computational Efficiency:")
    print(f"  Total GFLOP: {total_gflops:.2f}")
    print(f"  Total Runtime: {total_time:.2f} seconds")
    print(f"  GFLOP/sec: {gflops_per_sec:.2f}")
    print()
    
    print(f"Throughput:")
    print(f"  Samples per second: {samples_per_sec:.2f}")
    print(f"  Samples per batch: {n_samples}")
    print(f"  Time per sample: {(total_time/n_samples)*1000:.2f} ms")
    print()
    
    # GPU peak memory to FLOPS ratio
    if 'peak_memory_gb' in runtime_data:
        memory_efficiency = gflops_per_sec / runtime_data['peak_memory_gb']
        print(f"Memory Efficiency:")
        print(f"  GFLOP/sec per GB: {memory_efficiency:.2f}")
        print()
    
    return {
        'gflops_per_sec': gflops_per_sec,
        'samples_per_sec': samples_per_sec,
        'time_per_sample_ms': (total_time / n_samples) * 1000
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_results(gflops_data: Dict, runtime_data: Dict):
    """
    Create visualization of computational analysis
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: GFLOP Breakdown
    ax = axes[0, 0]
    components = ['Denoiser', 'FR Backbone', 'Cosine Sim', 'Voting']
    gflops = [
        gflops_data['denoiser_gflops'],
        gflops_data['fr_backbone_gflops'],
        gflops_data['cosine_similarity_gflops'],
        gflops_data['voting_gflops']
    ]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    ax.bar(components, gflops, color=colors)
    ax.set_ylabel('GFLOP', fontsize=10)
    ax.set_title('GFLOP Breakdown by Component', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: Runtime Breakdown
    ax = axes[0, 1]
    ax.bar(components, [g / gflops_data['total_gflops'] * 100 for g in gflops], color=colors)
    ax.set_ylabel('Percentage (%)', fontsize=10)
    ax.set_title('Computational Cost Distribution', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Per-Sample Time
    ax = axes[1, 0]
    time_per_sample = (runtime_data['avg_time_sec'] / gflops_data['n_samples']) * 1000
    ax.bar(['Per Sample'], [time_per_sample], color='#95E1D3')
    ax.set_ylabel('Time (ms)', fontsize=10)
    ax.set_title(f'Per-Sample Processing Time ({gflops_data["n_samples"]} samples)', 
                fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: GPU Efficiency
    ax = axes[1, 1]
    gflops_per_sec = gflops_data['total_gflops'] / runtime_data['avg_time_sec']
    metrics = ['GFLOP/sec', 'Samples/sec']
    values = [gflops_per_sec, gflops_data['n_samples'] / runtime_data['avg_time_sec']]
    ax.bar(metrics, values, color=['#667BC6', '#DA70D6'])
    ax.set_ylabel('Value', fontsize=10)
    ax.set_title('Performance Metrics', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Create the output directory if it doesn't exist
    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'visagekeeper_gflop_analysis.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}\n")
    
    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution: Calculate GFLOP and measure runtime
    """
    
    print("\n" + "="*80)
    print("VISAGEKEEPER ALGORITHM - GFLOP & GPU RUNTIME ANALYSIS")
    print("="*80)
    
    # ===== STEP 1: GFLOP CALCULATION =====
    gflops_data = calculate_total_pipeline_gflops(
        n_samples=N_SAMPLES,
        img_size=IMG_SIZE,
        embedding_dim=EMBEDDING_DIM,
        denoiser_channels=DENOISER_CHANNELS,
        denoiser_layers=DENOISER_LAYERS,
        model_type='simple'
    )
    
    # ===== STEP 2: COMPONENT RUNTIME MEASUREMENTS =====
    print("\n" + "="*80)
    print("COMPONENT-LEVEL RUNTIME MEASUREMENTS")
    print("="*80)
    
    denoiser_runtime = measure_denoiser_runtime(NUM_ITERATIONS)
    fr_runtime = measure_fr_backbone_runtime(NUM_ITERATIONS, model_type='simple')
    sim_runtime = measure_cosine_similarity_runtime(NUM_ITERATIONS * 10)
    
    # ===== STEP 3: FULL PIPELINE RUNTIME =====
    full_runtime = measure_full_pipeline_runtime(n_samples=N_SAMPLES, 
                                                n_test_iterations=10)
    
    # ===== STEP 4: GPU MEMORY ANALYSIS =====
    memory_data = analyze_gpu_memory(n_samples=100)
    
    # ===== STEP 5: PERFORMANCE METRICS =====
    performance_metrics = compute_performance_metrics(gflops_data, full_runtime)
    
    # ===== STEP 6: VISUALIZATION =====
    fig = visualize_results(gflops_data, full_runtime)
    
    # ===== STEP 7: GENERATE REPORT =====
    generate_report(gflops_data, denoiser_runtime, fr_runtime, sim_runtime, 
                   full_runtime, memory_data, performance_metrics)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE ✓")
    print("="*80 + "\n")

def generate_report(gflops_data, denoiser_rt, fr_rt, sim_rt, 
                   full_rt, memory_data, perf_metrics):
    """
    Generate comprehensive analysis report
    """
    
    report = f"""
{'='*80}
VISAGEKEEPER ALGORITHM - COMPREHENSIVE ANALYSIS REPORT
{'='*80}

HARDWARE INFORMATION:
  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
  CUDA Available: {torch.cuda.is_available()}
  {'GPU Memory: ' + f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB" if torch.cuda.is_available() else ""}

COMPUTATIONAL COMPLEXITY (GFLOP Analysis):
  Total GFLOP: {gflops_data['total_gflops']:.2f}
  Breakdown:
    - Gallery embedding: {gflops_data['gallery_gflops']:.4f} GFLOP
    - Denoiser ({gflops_data['n_samples']} samples): {gflops_data['denoiser_gflops']:.2f} GFLOP
    - FR Backbone ({gflops_data['n_samples']} samples): {gflops_data['fr_backbone_gflops']:.2f} GFLOP
    - Cosine similarity ({gflops_data['n_samples']} samples): {gflops_data['cosine_similarity_gflops']:.6f} GFLOP
    - Voting/Certification: {gflops_data['voting_gflops']:.6f} GFLOP

RUNTIME ANALYSIS:
  Full Pipeline:
    - Average: {full_rt['avg_time_sec']:.2f} ± {full_rt['std_time_sec']:.2f} seconds
    - Min/Max: {full_rt['min_time_sec']:.2f} / {full_rt['max_time_sec']:.2f} seconds
    - Per sample: {full_rt['per_sample_ms']:.2f} ms
  
  Component Times:
    - Denoiser: {denoiser_rt['avg_time_per_image_ms']:.2f} ms/image
    - FR Backbone: {fr_rt['avg_time_per_image_ms']:.2f} ms/image
    - Cosine similarity: {sim_rt['avg_time_per_pair_us']:.2f} μs/pair

GPU MEMORY USAGE:
  Model Parameters:
    - Denoiser: {memory_data.get('denoiser_params', 0):,} ({memory_data.get('denoiser_mb', 0):.2f} MB)
    - FR Model: {memory_data.get('fr_params', 0):,} ({memory_data.get('fr_mb', 0):.2f} MB)
  Peak Memory: {memory_data.get('peak_memory_gb', 0):.2f} GB

PERFORMANCE METRICS:
  GFLOP/sec: {perf_metrics['gflops_per_sec']:.2f}
  Samples/sec: {perf_metrics['samples_per_sec']:.2f}
  Time per sample: {perf_metrics['time_per_sample_ms']:.2f} ms

SCALABILITY:
  For n_samples = {gflops_data['n_samples']}:
    - Theoretical time (based on GFLOP): {gflops_data['total_gflops'] / 1000:.2f} seconds (at 1 TFLOP/s peak)
    - Actual time: {full_rt['avg_time_sec']:.2f} seconds
    - Efficiency: {(gflops_data['total_gflops'] / full_rt['avg_time_sec']) / 1000:.2f} TFLOP/s

COMPARISON WITH SAMURAI:
  SAMURAI: ~0.04 GFLOP per image
  VisageKeeper: {gflops_data['total_gflops'] / gflops_data['n_samples']:.4f} GFLOP per image (with denoising)

RECOMMENDATIONS:
  1. For real-time deployment (< 1 sec): Use n_samples = 100-200
  2. For high security (certified): Use n_samples >= 1000
  3. GFLOP/sec = {perf_metrics['gflops_per_sec']:.2f} suggests reasonable GPU utilization
  4. Memory usage = {memory_data.get('peak_memory_gb', 0):.2f} GB allows batch processing on most GPUs

{'='*80}
"""
    
    print(report)
    
    # Create the output directory if it doesn't exist
    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'visagekeeper_analysis_report.txt')
    
    # Save to file
    with open(save_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {save_path}\n")

if __name__ == '__main__':
    main()