# api_main.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import shutil
import json
from typing import List, Optional, Tuple
from pathlib import Path
import tempfile
import time
import uuid
import sys

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# ----------------------------------------------------------------------
# Configuration & Path Definitions
# ----------------------------------------------------------------------
print("--- Initializing Configuration and Paths ---")

SCRIPT_DIR = Path(__file__).resolve().parent
print(f"API Script Directory (SCRIPT_DIR): {SCRIPT_DIR}")

RELATIVE_PATH_TO_DATA_PIPELINE_DIR = "../Data_Pipeline"
RELATIVE_PATH_TO_MODEL_MODULE_DIR = "../Model"
RELATIVE_PATH_TO_MODEL_CHECKPOINT = "../Model/best_checkpoint.pth"

DATA_PIPELINE_MODULE_DIR = (SCRIPT_DIR / RELATIVE_PATH_TO_DATA_PIPELINE_DIR).resolve()
MODEL_SLGCN_MODULE_DIR = (SCRIPT_DIR / RELATIVE_PATH_TO_MODEL_MODULE_DIR).resolve()
DEFAULT_MODEL_CHECKPOINT_ABSOLUTE_PATH = (SCRIPT_DIR / RELATIVE_PATH_TO_MODEL_CHECKPOINT).resolve()

print(f"Expected Data Pipeline Module Directory (absolute): {DATA_PIPELINE_MODULE_DIR}")
print(f"Expected SLGCN Module Directory (absolute): {MODEL_SLGCN_MODULE_DIR}")
print(f"Expected Model Checkpoint File (absolute): {DEFAULT_MODEL_CHECKPOINT_ABSOLUTE_PATH}")

TARGET_FRAMES_PER_VIDEO_CONFIG = 30
# CRITICAL: Ensure this matches the complexity used in your original script for fair comparison
MEDIAPIPE_MODEL_COMPLEXITY_CONFIG = 1 # Options: 0 (fastest), 1 (default), 2 (most accurate)
# ----------------------------------------------------------------------
# End of Configuration & Path Definitions
# ----------------------------------------------------------------------

# --- Attempt to import IN-MEMORY custom module functions ---
print("--- Attempting to import custom modules for IN-MEMORY processing ---")
try:
    if str(DATA_PIPELINE_MODULE_DIR) not in sys.path:
        sys.path.append(str(DATA_PIPELINE_MODULE_DIR))
        print(f"Added to sys.path: {DATA_PIPELINE_MODULE_DIR}")
    if str(MODEL_SLGCN_MODULE_DIR) not in sys.path:
        sys.path.append(str(MODEL_SLGCN_MODULE_DIR))
        print(f"Added to sys.path: {MODEL_SLGCN_MODULE_DIR}")

    print("Importing S01_Frame_Extractor.extract_frames_to_memory...")
    # Ensure S01_Frame_Extractor.py has this function and it uses frame.copy() if needed
    from S01_Frame_Extractor import extract_frames_to_memory
    print("Importing from S02_Keypoint_Extractor (in-memory version)...")
    # Ensure S02_Keypoint_Extractor.py has this function
    from S02_Keypoint_Extractor import (
        extract_keypoints_from_memory_frames,
        TOTAL_LANDMARKS_TO_EXTRACT,
        DIMENSIONS_PER_LANDMARK
    )
    print("Importing SLGCN.DecoupledGCN...")
    from SLGCN import DecoupledGCN
    print("Custom modules for in-memory processing imported successfully.")

except ImportError as e:
    print(f"ERROR: Critical error importing necessary modules for in-memory processing: {e}")
    print("Please ensure S01_Frame_Extractor.py contains 'extract_frames_to_memory' (with frame.copy() if needed) and S02_Keypoint_Extractor.py contains 'extract_keypoints_from_memory_frames'.")
    exit(1)
except Exception as e:
    print(f"ERROR: An unexpected error occurred during module import setup: {e}")
    exit(1)
print("--- Module imports complete ---")

# --- Model Configuration ---
GRAPH_ARGS_CONFIG = {
    'num_nodes': TOTAL_LANDMARKS_TO_EXTRACT,
    'inward_edges': [[2,0],[1,0],[0,3],[0,4],[3,5],[4,6],[5,7],[6,17],[7,8],[7,9],[9,10],[7,11],[11,12],[7,13],[13,14],[7,15],[15,16],[17,18],[17,19],[19,20],[17,21],[21,22],[17,23],[23,24],[17,25],[25,26]]
}
INPUT_CHANNELS_CONFIG = DIMENSIONS_PER_LANDMARK
MODEL_N_OUT_FEATURES_CONFIG = 256

# --- Global variables ---
model_instance: Optional[nn.Module] = None
gloss_map_instance: Optional[List[str]] = None
device_instance: Optional[torch.device] = None

app = FastAPI(title="Sign Language Gloss Prediction API (In-Memory Frame Processing with Profiling)")

# --- Preprocessing Function (In-Memory) with Granular Profiling ---
def preprocess_video_to_keypoints_in_memory_for_api(
    uploaded_video_path: str, # Path to the temporarily saved uploaded video
    target_frames: int,
    mp_model_complexity: int
) -> Optional[np.ndarray]:
    print(f"  Starting IN-MEMORY preprocessing for video: {Path(uploaded_video_path).name}")
    overall_preprocess_start_time = time.perf_counter()
    keypoints_data = None

    try:
        # --- Profiling S01 (In-Memory Frame Extraction) ---
        time_s01_start = time.perf_counter()
        print(f"    Calling S01_Frame_Extractor.extract_frames_to_memory...")
        frames_list: Optional[List[np.ndarray]] = extract_frames_to_memory(
            video_path=uploaded_video_path,
            target_frames=target_frames
        )
        time_s01_end = time.perf_counter()
        print(f"    PROFILING: S01_Frame_Extractor (extract_frames_to_memory) took: {time_s01_end - time_s01_start:.4f} seconds.")

        if frames_list is None or not frames_list:
            print(f"    ERROR: In-memory frame extraction failed for {uploaded_video_path}.")
            return None
        print(f"    Frames extracted to memory successfully: {len(frames_list)} frames.")

        # --- Profiling S02 (In-Memory Keypoint Extraction) ---
        time_s02_start = time.perf_counter()
        print(f"    Calling S02_Keypoint_Extractor.extract_keypoints_from_memory_frames...")
        keypoints_data = extract_keypoints_from_memory_frames(
            frames_list=frames_list,
            target_frames_count=target_frames, # This should match the number of frames S01 prepared
            model_complexity_setting=mp_model_complexity
        )
        time_s02_end = time.perf_counter()
        print(f"    PROFILING: S02_Keypoint_Extractor (extract_keypoints_from_memory_frames) took: {time_s02_end - time_s02_start:.4f} seconds.")
        
        del frames_list # Free memory once keypoints are extracted

        if keypoints_data is None:
            print(f"    ERROR: Keypoint extraction from memory failed.")
            return None
        print(f"    Keypoints extracted from memory successfully. Shape: {keypoints_data.shape}")

    except Exception as e:
        import traceback
        print(f"    ERROR: An unexpected error occurred during IN-MEMORY preprocessing for {uploaded_video_path}: {e}")
        traceback.print_exc()
        return None
    finally:
        overall_preprocess_end_time = time.perf_counter()
        print(f"    PROFILING: Total preprocess_video_to_keypoints_in_memory_for_api took: {overall_preprocess_end_time - overall_preprocess_start_time:.4f} seconds.")
    
    return keypoints_data

# --- Prediction Function with Granular Profiling ---
def predict_gloss_from_keypoints_api_adapt(
    keypoints_np: np.ndarray, model: nn.Module, gloss_map: List[str], device: torch.device,
) -> Optional[Tuple[str, float]]:
    time_tensor_prep_start = time.perf_counter()
    keypoints_tensor = torch.from_numpy(keypoints_np).float().permute(2, 0, 1)
    keypoints_tensor = keypoints_tensor.unsqueeze(0).to(device)
    time_tensor_prep_end = time.perf_counter()
    print(f"    PROFILING: Tensor preparation took: {time_tensor_prep_end - time_tensor_prep_start:.4f} seconds.")

    time_model_infer_start = time.perf_counter()
    print("    Running model inference...")
    model.eval()
    with torch.no_grad():
        features = model(keypoints_tensor, keep_prob=1.0) 
        outputs = model.classifier(features) if model.classifier else None
    time_model_infer_end = time.perf_counter()
    print(f"    PROFILING: Actual GCN model inference took: {time_model_infer_end - time_model_infer_start:.4f} seconds.")

    if outputs is None:
        print("    ERROR: Model classifier is None or inference failed before output generation.")
        return None

    time_post_process_start = time.perf_counter()
    probabilities = F.softmax(outputs, dim=1)
    confidence, predicted_idx_tensor = torch.max(probabilities, 1)
    predicted_idx = predicted_idx_tensor.item()
    confidence_score = confidence.item()
    time_post_process_end = time.perf_counter()
    print(f"    PROFILING: Output post-processing (softmax, max) took: {time_post_process_end - time_post_process_start:.4f} seconds.")

    if predicted_idx >= len(gloss_map):
        print(f"    ERROR: Predicted index {predicted_idx} is out of bounds for gloss_map with size {len(gloss_map)}.")
        return None
    predicted_gloss = gloss_map[predicted_idx]
    print(f"    Prediction successful: {predicted_gloss}")
    return predicted_gloss, confidence_score

# --- FastAPI Event Handler: Model Loading on Startup ---
@app.on_event("startup")
async def load_model_on_startup():
    global model_instance, gloss_map_instance, device_instance
    print("--- API Startup: Loading Model ---")
    device_instance = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device_instance}")
    checkpoint_file_path = DEFAULT_MODEL_CHECKPOINT_ABSOLUTE_PATH
    print(f"Attempting to load model checkpoint from: {checkpoint_file_path}")

    if not checkpoint_file_path.exists():
        print(f"FATAL ERROR: Model checkpoint not found at {checkpoint_file_path}")
        return
    try:
        checkpoint = torch.load(checkpoint_file_path, map_location=device_instance)
        print(f"Checkpoint loaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Error loading checkpoint: {e}")
        return

    model_state_dict = checkpoint.get('model_state_dict')
    selected_glosses = checkpoint.get('selected_glosses')

    if model_state_dict is None:
        if isinstance(checkpoint, dict) and not any(k in checkpoint for k in ['model_state_dict', 'selected_glosses']):
            model_state_dict = checkpoint
            print("WARNING: Loaded checkpoint appears to be only a model_state_dict.")
        else:
            print("FATAL ERROR: Checkpoint is missing 'model_state_dict'.")
            return
    if selected_glosses is None:
        print("FATAL ERROR: 'selected_glosses' not found in checkpoint.")
        return

    gloss_map_instance = selected_glosses
    num_classes = len(gloss_map_instance)
    print(f"Number of classes from gloss map: {num_classes}")

    try:
        current_model = DecoupledGCN(
            in_channels=INPUT_CHANNELS_CONFIG,
            graph_args=GRAPH_ARGS_CONFIG,
            n_out_features=MODEL_N_OUT_FEATURES_CONFIG
        )
        current_model.classifier = nn.Linear(current_model.n_out_features, num_classes)
        current_model.to(device_instance)
        current_model.eval()
        current_model.load_state_dict(model_state_dict)
        model_instance = current_model
        print("Model initialized and state loaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Error initializing or loading model state: {e}")
        return
    print("--- Model Loading Complete ---")

# --- FastAPI Endpoint: Video Prediction ---
@app.post("/predict/", response_class=JSONResponse)
async def predict_video_gloss(file: UploadFile = File(...)):
    global model_instance, gloss_map_instance, device_instance
    if not model_instance or not gloss_map_instance or not device_instance:
        raise HTTPException(status_code=503, detail="Model not loaded or unavailable. Check server logs.")

    print(f"\n--- Received request for video: {file.filename} ---")
    request_overall_start_time = time.perf_counter()
    
    temp_uploaded_video_path_str: Optional[str] = None
    keypoints_np = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix if file.filename else ".mp4") as tf:
            time_upload_save_start = time.perf_counter()
            shutil.copyfileobj(file.file, tf)
            temp_uploaded_video_path_str = tf.name 
            time_upload_save_end = time.perf_counter()
            print(f"  Uploaded video content temporarily available at: {temp_uploaded_video_path_str}")
            print(f"  PROFILING: Saving uploaded video took: {time_upload_save_end - time_upload_save_start:.4f} seconds.")
        
        keypoints_np = preprocess_video_to_keypoints_in_memory_for_api(
            uploaded_video_path=temp_uploaded_video_path_str,
            target_frames=TARGET_FRAMES_PER_VIDEO_CONFIG,
            mp_model_complexity=MEDIAPIPE_MODEL_COMPLEXITY_CONFIG
        )

        if keypoints_np is None:
            print(f"  Prediction failed due to in-memory preprocessing error.")
            raise HTTPException(status_code=422, detail="Video preprocessing (in-memory) failed.")

        prediction_result = predict_gloss_from_keypoints_api_adapt(
            keypoints_np=keypoints_np,
            model=model_instance,
            gloss_map=gloss_map_instance,
            device=device_instance
        )
        del keypoints_np

    except Exception as e:
        import traceback
        print(f"  Error during request processing: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error during video processing: {e}")
    finally:
        file.file.close()
        if temp_uploaded_video_path_str and Path(temp_uploaded_video_path_str).exists():
            try:
                os.remove(temp_uploaded_video_path_str)
                print(f"  Cleaned up temporary uploaded video file: {temp_uploaded_video_path_str}")
            except Exception as e_del:
                print(f"  Warning: Could not delete temporary uploaded video file {temp_uploaded_video_path_str}: {e_del}")
    
    request_overall_end_time = time.perf_counter()
    total_server_processing_time = request_overall_end_time - request_overall_start_time
    print(f"  PROFILING: Total server-side processing for request: {total_server_processing_time:.4f} seconds.")

    if prediction_result:
        predicted_gloss, confidence = prediction_result
        return {
            "filename": file.filename,
            "predicted_gloss": predicted_gloss,
            "confidence": round(confidence, 4),
            "prediction_time_seconds": round(total_server_processing_time, 4)
        }
    else:
        print(f"  Failed to obtain a prediction for {file.filename}.")
        raise HTTPException(status_code=500, detail="Model inference failed after preprocessing or preprocessing itself failed.")

if __name__ == "__main__":
    print(f"To run the API, use Uvicorn from your terminal:")
    print(f"Navigate to '{SCRIPT_DIR}' directory (where this api_main.py file is located).")
    print(f"Then run: uvicorn {Path(__file__).stem}:app --reload --host 0.0.0.0 --port 8000")