#!/usr/bin/env python3
"""
Find Weight Files in VisageKeeper Project
Recursively searches for .pt, .pkl, .pth files
Also inspects directory structure
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

def find_weight_files(root_dir, max_depth=5):
    """Recursively find weight files"""
    weight_extensions = ['.pt', '.pth', '.pkl', '.pickle', '.bin', '.onnx']
    found_files = defaultdict(list)
    
    print(f"\n{'='*80}")
    print(f"SEARCHING FOR WEIGHT FILES: {root_dir}")
    print(f"{'='*80}\n")
    
    if not os.path.isdir(root_dir):
        print(f"✗ Directory not found: {root_dir}")
        return found_files
    
    for root, dirs, files in os.walk(root_dir):
        # Limit depth
        depth = root.replace(root_dir, '').count(os.sep)
        if depth > max_depth:
            continue
        
        for file in files:
            if any(file.endswith(ext) for ext in weight_extensions):
                filepath = os.path.join(root, file)
                file_size = os.path.getsize(filepath)
                ext = Path(file).suffix
                
                found_files[ext].append({
                    'path': filepath,
                    'name': file,
                    'size': file_size,
                    'size_mb': file_size / (1024*1024)
                })
    
    return found_files

def print_directory_structure(root_dir, max_items=20, max_depth=2):
    """Print directory tree structure"""
    print(f"\n{'='*80}")
    print(f"DIRECTORY STRUCTURE: {root_dir}")
    print(f"{'='*80}\n")
    
    if not os.path.isdir(root_dir):
        print(f"✗ Directory not found: {root_dir}")
        return
    
    items = os.listdir(root_dir)
    dirs = [i for i in items if os.path.isdir(os.path.join(root_dir, i))]
    files = [i for i in items if os.path.isfile(os.path.join(root_dir, i))]
    
    print(f"📁 Directories: {len(dirs)}")
    for d in dirs[:max_items]:
        subdir_path = os.path.join(root_dir, d)
        subdir_items = os.listdir(subdir_path)
        subdir_count = len(subdir_items)
        print(f"   └─ {d}/ ({subdir_count} items)")
    if len(dirs) > max_items:
        print(f"   ... and {len(dirs) - max_items} more directories")
    
    print(f"\n📄 Files: {len(files)}")
    for f in files[:max_items]:
        file_path = os.path.join(root_dir, f)
        file_size = os.path.getsize(file_path) / (1024*1024)
        print(f"   └─ {f} ({file_size:.2f} MB)")
    if len(files) > max_items:
        print(f"   ... and {len(files) - max_items} more files")

def main():
    print("\n" + "="*80)
    print("VisageKeeper Weight File Finder")
    print("="*80)
    
    # Search in common locations
    search_paths = [
        os.path.expanduser("~/Deepraj/Paper Codes/VisageKeeper"),
        os.path.expanduser("~/Deepraj/Paper Codes/VisageKeeper/results"),
        os.path.expanduser("~/Deepraj/Paper Codes/VisageKeeper/checkpoints"),
        os.path.expanduser("~/Deepraj/Paper Codes/VisageKeeper/models"),
        os.path.expanduser("~/Deepraj/Paper Codes/VisageKeeper/section_codes"),
        os.getcwd(),
    ]
    
    print("\nSearching in:")
    for path in search_paths:
        if os.path.exists(path):
            print(f"  ✓ {path}")
        else:
            print(f"  ✗ {path} (not found)")
    
    all_found = {}
    
    # Search each path
    for base_path in search_paths:
        if os.path.exists(base_path):
            found = find_weight_files(base_path, max_depth=4)
            if found:
                all_found[base_path] = found
            
            # Also show directory structure
            print_directory_structure(base_path, max_items=15)
    
    # Print summary
    print("\n" + "="*80)
    print("WEIGHT FILES FOUND")
    print("="*80 + "\n")
    
    if not all_found:
        print("✗ No weight files found!")
        print("\nCommon places to check:")
        print("  1. ~/Deepraj/Paper Codes/VisageKeeper/results/")
        print("  2. ~/Deepraj/Paper Codes/VisageKeeper/checkpoints/")
        print("  3. ~/Deepraj/Paper Codes/VisageKeeper/models/")
        print("  4. ~/Deepraj/Paper Codes/VisageKeeper/ (root)")
        print("\nLooking for denoiser weights specifically:")
        print("  - denoiser.pt / denoiser.pth")
        print("  - denoiser_s050.pth")
        print("  - model.pt / model.pth")
        return 1
    
    total_size = 0
    for base_path, found in all_found.items():
        print(f"\n📍 {base_path}")
        print("-" * 80)
        
        for ext, files in sorted(found.items()):
            print(f"\n  {ext} files ({len(files)}):")
            
            for file_info in sorted(files, key=lambda x: x['size'], reverse=True):
                rel_path = file_info['path'].replace(base_path, "").lstrip("/")
                print(f"    • {rel_path}")
                print(f"      Size: {file_info['size_mb']:.2f} MB")
                total_size += file_info['size']
    
    print("\n" + "="*80)
    print(f"✓ TOTAL SIZE: {total_size / (1024*1024):.2f} MB")
    print("="*80 + "\n")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
