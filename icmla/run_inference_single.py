"""
Script to run Qwen2VL on random input images.
Generates synthetic random images and runs inference.

Command to run:
python icmla/run_inference_single.py
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import tempfile
import os
from pathlib import Path
import time
import shutil
import glob
from prompts import prompt_percep_action, prompts_general_action_gen, prompts_general_action_gen_overwrite, prompts_add_ped_crossing

# Optional: Set local model cache directory (None = use default ~/.cache/huggingface)
# Change this to save models to a custom location
LOCAL_MODEL_CACHE = "/Users/ethangao/icmla/models"  # Use default cache

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

FILE_LIST = [
    # "ce80eb1d-67637c6c.jpg",
    # "cea0f571-99895ca1.jpg",
    # "d1ef7d87-00000000.jpg",
    # "d53c7574-00000000.jpg",
    # "d6c78174-00000000.jpg",
    # "e2849bb0-ab67126b.jpg",
    # "e28c2e57-00000000.jpg"
    "c20d8eaf-2e41fdc8.jpg",
    "b94a9ed0-081117e5.jpg",
    "cb7c8bb9-f3e44cc5.jpg",
    "d7402ea7-cc563c45.jpg",
    "d748c348-74219f6b.jpg",
    "d795d9d8-20387be2.jpg",
    "da1acd1f-a7d03845.jpg",
    "dc81e22b-a10379b4.jpg"
    
]

def setup_model_cache():
    """Setup local model cache if specified."""
    if LOCAL_MODEL_CACHE:
        os.environ["HF_HOME"] = LOCAL_MODEL_CACHE
        Path(LOCAL_MODEL_CACHE).mkdir(parents=True, exist_ok=True)
        print(f"Using model cache: {LOCAL_MODEL_CACHE}")

# Load model and processor
setup_model_cache()
print("Loading Qwen2-VL model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)


def load_all_images_from_dataset_ninja(dataset_path=None):
    """Load all images from specified dataset path or dataset-ninja folder."""
    if dataset_path is None:
        dataset_path = Path.home() / "dataset-ninja"
    else:
        dataset_path = Path(dataset_path)
    
    # Find image files
    image_patterns = ['**/*.jpg', '**/*.jpeg', '**/*.png']
    image_files = []
    for pattern in image_patterns:
        # image_files.extend(list(dataset_path.glob(pattern)))
        for file in dataset_path.glob(pattern):
            print(f"Checking file: {file.name}")
            if file.name in FILE_LIST:
                image_files.append(file)
                print(f"  ✓ Found image: {file.name}")
    
    if not image_files:
        print(f"No images found in {dataset_path}")
        return []
    
    print(f"Found {len(image_files)} images in {dataset_path}")
    return sorted(image_files)  # Sort for consistent ordering

def run_inference(image, prompt):
    """Run Qwen2VL inference on an image with given prompt."""
    start_time = time.time()
    
    # Save image temporarily
    print("  [1/5] Saving image...")
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        image.save(tmp.name)
        image_path = tmp.name
    
    try:
        print("  [2/5] Processing message...")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        print("  [3/5] Applying chat template...")
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        print("  [4/5] Extracting vision info...")
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        inputs = inputs.to(model.device)
        
        print("  [5/5] Generating response (this may take 1-2 minutes on CPU)...")
        gen_start = time.time()
        try:
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=False,  # Use greedy decoding for stability
                temperature=1.0
            )
        except RuntimeError as e:
            if "inf" in str(e) or "nan" in str(e):
                print(f"    ⚠ Numerical stability issue: {e}")
                print("    Retrying with adjusted parameters...")
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    temperature=0.7
                )
            else:
                raise
        
        gen_time = time.time() - gen_start
        
        output = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        total_time = time.time() - start_time
        print(f"  ✓ Inference complete! (Generation: {gen_time:.1f}s, Total: {total_time:.1f}s)")
        
        return output[0]
    finally:
        os.unlink(image_path)

if __name__ == "__main__":
    prompts = """
    You are an autonomous driving perception system.

Analyze the driving scene and extract all information relevant to driving safety.

Output JSON only.

{
  "weather": "",
  "time_of_day": "",
  "road_type": "",
  "intersection": "",
  "lane_count": "",
  "traffic_density": "",
  "vehicles": [],
  "pedestrians": [],
  "cyclists": [],
  "traffic_lights": [],
  "traffic_signs": [],
  "road_conditions": [],
  "potential_hazards": [],
  "overall_risk_level": ""
}

Rules:
- Use concise values.
- Do not include explanations.
- Return valid JSON only."""
    prompts = """You are an autonomous driving perception system.

Analyze the driving scene and extract ONLY visually observable information.

Output JSON only.

CRITICAL RULES:
- Output valid JSON only.
- Never use "", null, or None.
- If unknown → use "unknown" (for strings) or [] (for lists).
- Do not guess or infer missing attributes.
- Only report what is directly visible in the image.

FIELDS:
{
  "weather": "unknown",
  "time_of_day": "unknown",
  "road_type": "unknown",
  "intersection": "unknown",
  "lane_count": "unknown",
  "traffic_density": "unknown",
  "vehicles": [],
  "pedestrians": [],
  "cyclists": [],
  "traffic_lights": [],
  "traffic_signs": [],
  "road_conditions": [],
  "potential_hazards": [],
  "overall_risk_level": "unknown"
}"""

    # prompts = prompts_general_action_gen
    prompts =prompts_add_ped_crossing
    
    
    print("\n" + "="*60)
    print("Running Qwen2-VL on all samples from BDD100K test set")
    print("="*60)
    
    # Setup results directory
    results_dir = Path("/Users/ethangao/icmla/inference_results_exp_counterfactual")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load images from specific directory
    dataset_path = "/Users/ethangao/dataset-ninja/bdd100k:-images-10k/test/img"
    print(f"\nLoading images from {dataset_path}...")
    image_files = load_all_images_from_dataset_ninja(dataset_path)
    
    if not image_files:
        print(f"❌ No images found in {dataset_path}")
        exit(1)
    
    # Create main results file
    results_file = results_dir / f"results_7_{len(image_files)}_samples_p4_add_ped_crossing.txt"
    
    with open(results_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"Qwen2-VL Inference Results (All {len(image_files)} Samples from BDD100K)\n")
        f.write("="*60 + "\n\n")
    
    # Run all samples
    num_samples = len(image_files)
    successful = 0
    failed = 0
    
    file_list = set(FILE_LIST)
    
    for sample_idx in range(num_samples):
        print(f"\n[Sample {sample_idx + 1}/{num_samples}]")
        print("-" * 40)
        
        image_path = image_files[sample_idx]
        image_filename = image_path.name
        
        # if image_filename not in file_list:
        #     print(f"  Skipping {image_filename} (not in FILE_LIST)")
        #     continue
        
        print(f"  Loading image: {image_filename}")
        
        try:
            # Load image
            sample_image = Image.open(image_path)
            if sample_image.mode != 'RGB':
                sample_image = sample_image.convert('RGB')
            
            # Use rotating prompts
            prompt = prompts
            print(f"  Prompt: {prompt}")
            
            # Run inference
            response = run_inference(sample_image, prompt)
            print(f"  Response: {response[:100]}...")
            
            # Append to results file
            with open(results_file, 'a') as f:
                f.write(f"Sample {sample_idx + 1}:\n")
                f.write(f"  Image: {image_filename}\n")
                f.write(f"  Path: {image_path}\n")
                # f.write(f"  Prompt: {prompt}\n")
                f.write(f"  Response: {response}\n")
                f.write("\n" + "-"*60 + "\n\n")
            
            successful += 1
            print(f"  ✓ Sample {sample_idx + 1} completed successfully")
            
        except Exception as e:
            failed += 1
            print(f"  ❌ Sample {sample_idx + 1} failed: {e}")
            
            with open(results_file, 'a') as f:
                f.write(f"Sample {sample_idx + 1}:\n")
                f.write(f"  Image: {image_filename}\n")
                f.write(f"  Path: {image_path}\n")
                f.write(f"  Prompt: {prompt}\n")
                f.write(f"  Response: ERROR - {str(e)}\n")
                f.write("\n" + "-"*60 + "\n\n")
    
    # Summary
    print("\n" + "="*60)
    print(f"Results Summary: {successful} successful, {failed} failed")
    print("="*60)
    print(f"✓ All results saved to: {results_file}\n")
