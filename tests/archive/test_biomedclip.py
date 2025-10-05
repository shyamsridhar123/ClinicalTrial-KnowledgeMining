"""Test BiomedCLIP for clinical trial multimodal embeddings."""

import torch
import time
from pathlib import Path
from PIL import Image
import json

def test_biomedclip_feasibility():
    """Test if BiomedCLIP can handle our clinical trial content."""
    
    print("=== BiomedCLIP Feasibility Test ===")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        print(f"VRAM Free: {torch.cuda.memory_reserved(0) / 1024**3:.1f}GB")
    
    try:
        # Test model loading
        print("\n1. Testing model availability...")
        from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
        
        model_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        
        start_time = time.time()
        print(f"Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"Loading image processor...")
        image_processor = AutoImageProcessor.from_pretrained(model_name)
        
        print(f"Loading model...")
        model = AutoModel.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            model = model.cuda()
            print("Model moved to GPU")
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.1f}s")
        
        # Check model memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_used = torch.cuda.memory_allocated(0) / 1024**3
            print(f"GPU Memory Used: {memory_used:.2f}GB")
        
        # Test with clinical trial text
        print("\n2. Testing text embedding...")
        clinical_text = "Figure 4 shows box and whiskers plot of efficacy endpoint measurements comparing treatment groups. The primary endpoint demonstrated statistical significance (p<0.05) with median improvement in the active arm."
        
        start_time = time.time()
        text_inputs = tokenizer(clinical_text, return_tensors="pt", padding=True, truncation=True)
        if torch.cuda.is_available():
            text_inputs = {k: v.cuda() for k, v in text_inputs.items()}
        
        with torch.no_grad():
            text_outputs = model.get_text_features(**text_inputs)
        
        text_time = time.time() - start_time
        print(f"✅ Text embedding: {text_outputs.shape} in {text_time:.3f}s")
        
        # Test with a sample figure if available
        print("\n3. Testing image embedding...")
        figure_path = Path("data/processing/figures/DV07_test/DV07/DV07_figure_01.png")
        
        if figure_path.exists():
            print(f"Loading figure: {figure_path}")
            image = Image.open(figure_path).convert('RGB')
            print(f"Image size: {image.size}")
            
            start_time = time.time()
            image_inputs = image_processor(images=image, return_tensors="pt")
            if torch.cuda.is_available():
                image_inputs = {k: v.cuda() for k, v in image_inputs.items()}
            
            with torch.no_grad():
                image_outputs = model.get_image_features(**image_inputs)
            
            image_time = time.time() - start_time
            print(f"✅ Image embedding: {image_outputs.shape} in {image_time:.3f}s")
            
            # Test similarity
            print("\n4. Testing text-image similarity...")
            similarity = torch.cosine_similarity(text_outputs, image_outputs)
            print(f"Text-Image Similarity: {similarity.item():.4f}")
            
        else:
            print(f"❌ Sample figure not found at {figure_path}")
            
        # Performance summary
        print(f"\n=== PERFORMANCE SUMMARY ===")
        print(f"Model Load Time: {load_time:.1f}s")
        print(f"Text Inference: {text_time*1000:.1f}ms")
        if figure_path.exists():
            print(f"Image Inference: {image_time*1000:.1f}ms")
        if torch.cuda.is_available():
            print(f"GPU Memory Usage: {memory_used:.2f}GB / 4.0GB")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Try: pixi add transformers pillow torch torchvision")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_biomedclip_feasibility()
