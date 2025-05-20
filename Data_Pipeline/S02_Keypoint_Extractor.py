import os
import cv2
import numpy as np
import mediapipe as mp
from scipy import interpolate
from pathlib import Path
from typing import List, Optional, Tuple

# --- Constants ---
DEFAULT_TARGET_FRAMES = 30
DEFAULT_MODEL_COMPLEXITY = 1

UPPER_POSE_LANDMARK_INDICES = list(range(23))
NUM_UPPER_POSE_LANDMARKS = len(UPPER_POSE_LANDMARK_INDICES)
NUM_HAND_LANDMARKS = 21
TOTAL_LANDMARKS_65 = NUM_UPPER_POSE_LANDMARKS + NUM_HAND_LANDMARKS + NUM_HAND_LANDMARKS
DIMENSIONS_PER_LANDMARK = 3

LEFT_SHOULDER_IDX_IN_65 = 11
RIGHT_SHOULDER_IDX_IN_65 = 12

SELECTED_INDICES_FROM_65_TO_27 = [
    0, 2, 5, 11, 12, 13, 14,
    23+0, 23+4, 23+5, 23+8, 23+9, 23+12, 23+13, 23+16, 23+17, 23+20,
    44+0, 44+4, 44+5, 44+8, 44+9, 44+12, 44+13, 44+16, 44+17, 44+20
]
TOTAL_LANDMARKS_TO_EXTRACT = len(SELECTED_INDICES_FROM_65_TO_27) # Should be 27

# --- Helper Functions (Aligned with user's Colab script logic) ---

def load_and_check_frames(instance_frames_dir_str: str, target_frame_count: int) -> Optional[List[np.ndarray]]:
    instance_dir = Path(instance_frames_dir_str)
    if not instance_dir.is_dir():
        print(f"  Helper Error: Instance directory not found: {instance_frames_dir_str}")
        return None
    frames = []
    frame_files = sorted(list(instance_dir.glob('*.png')) + list(instance_dir.glob('*.jpg')) + list(instance_dir.glob('*.jpeg')))
    if len(frame_files) != target_frame_count:
        # This can be a common case if S01 padding occurred for a short video.
        # The Colab logic often assumes frames are already there and matches target_frame_count exactly.
        # For robustness, we might allow fewer frames if padding happened in S01 and S02 can handle fewer.
        # However, to strictly match, this check is kept.
        print(f"  Helper Warning: Instance {instance_dir.name}. Expected {target_frame_count} frames, found {len(frame_files)}.")
        # If we want to be strict and fail:
        return None
    for frame_file_path in frame_files:
        frame = cv2.imread(str(frame_file_path))
        if frame is not None: frames.append(frame)
        else:
            print(f"  Helper Warning: Instance {instance_dir.name}. Failed to load frame: {frame_file_path.name}")
            return None
    return frames

def interpolate_missing_keypoints(keypoints_array: np.ndarray) -> Optional[np.ndarray]:
    if keypoints_array is None or keypoints_array.shape[0] == 0: return keypoints_array
    frames, num_landmarks, dims = keypoints_array.shape
    interpolated_data = keypoints_array.copy()
    for landmark_idx in range(num_landmarks):
        for dim_idx in range(dims):
            time_series = interpolated_data[:, landmark_idx, dim_idx]
            valid_indices = np.where(time_series != 0)[0]
            if len(valid_indices) == frames: continue
            if len(valid_indices) < 2: continue
            try:
                interp_func = interpolate.interp1d(valid_indices, time_series[valid_indices], kind='linear', bounds_error=False, fill_value=(time_series[valid_indices[0]], time_series[valid_indices[-1]]))
                interpolated_data[:, landmark_idx, dim_idx] = interp_func(np.arange(frames))
            except ValueError:
                pass
    return interpolated_data

def normalize_keypoints(keypoints_array: np.ndarray, left_shoulder_idx: int, right_shoulder_idx: int, num_pose_landmarks: int) -> Optional[np.ndarray]:
    if keypoints_array is None or keypoints_array.shape[0] == 0: return keypoints_array
    frames, total_landmarks, dims = keypoints_array.shape
    normalized_data = keypoints_array.copy()
    if not (0 <= left_shoulder_idx < num_pose_landmarks and 0 <= right_shoulder_idx < num_pose_landmarks and left_shoulder_idx < total_landmarks and right_shoulder_idx < total_landmarks):
        print(f"  Helper Error: Invalid shoulder indices for normalization. L:{left_shoulder_idx}, R:{right_shoulder_idx}, PoseCount:{num_pose_landmarks}, Total:{total_landmarks}")
        return None
    for i in range(frames):
        frame_kps = normalized_data[i, :, :]
        l_sh = frame_kps[left_shoulder_idx, :]
        r_sh = frame_kps[right_shoulder_idx, :]
        if np.all(l_sh == 0) and np.all(r_sh == 0):
            continue
        center = (l_sh + r_sh) / 2.0
        shoulder_diff = l_sh - r_sh
        scale = np.linalg.norm(shoulder_diff)
        epsilon = 1e-6
        if scale < epsilon:
            print(f"    Normalize WARNING: Frame {i}, very small shoulder scale ({scale:.2e}). Setting frame keypoints to 0.")
            normalized_data[i, :, :] = 0.0
            continue
        normalized_data[i, :, :] = (frame_kps - center) / scale
    return normalized_data

def _extract_raw_65_keypoints_from_frames(
    frames_list: List[np.ndarray],
    model_complexity_setting: int,
    target_frames: int # Renamed from target_frames to avoid confusion with len(frames_list)
) -> Optional[np.ndarray]:
    if not frames_list:
        print("  _extract_raw_65... Error: Empty frames_list provided.")
        return None
    
    if len(frames_list) != target_frames:
        print(f"  _extract_raw_65... Warning: Number of frames in list ({len(frames_list)}) does not match target_frames ({target_frames}). Proceeding with actual number of frames.")
        # This might happen if S01 padding logic changes or is bypassed.
        # The expected shape will be based on actual frames in list for robustness.
        # However, the caller (especially the in-memory version) should ensure consistency.
        # For strictness with `target_frames` being the desired output dimension:
        # If the expectation is that `frames_list` *always* has `target_frames` elements (due to S01 padding),
        # then this check could be an error. Assuming S01 provides exactly `target_frames`.
        # For now, we'll use the provided `target_frames` for the *expected output shape*
        # and `len(frames_list)` for iteration.
    
    # The output array should always have `target_frames` as its first dimension if S01 guarantees padding.
    expected_shape_65 = (target_frames, TOTAL_LANDMARKS_65, DIMENSIONS_PER_LANDMARK)

    mp_holistic = mp.solutions.holistic
    holistic_model = mp_holistic.Holistic(
        static_image_mode=False, # Process as a video stream
        model_complexity=model_complexity_setting,
        enable_segmentation=False
    )
    all_frames_keypoints_65 = []

    for frame_idx, frame_img in enumerate(frames_list):
        if frame_idx >= target_frames: # Ensure we don't process more frames than intended for output
            print(f"  _extract_raw_65... Warning: Processing stopped at frame {frame_idx} as target_frames ({target_frames}) reached.")
            break

        image_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        try:
            results = holistic_model.process(image_rgb)
        except Exception as e:
            print(f"    WARNING: MediaPipe processing error on frame {frame_idx}: {e}. Using zeros.")
            all_frames_keypoints_65.append(np.zeros((TOTAL_LANDMARKS_65, DIMENSIONS_PER_LANDMARK), dtype=np.float32))
            continue
        
        current_frame_keypoints_65 = np.zeros((TOTAL_LANDMARKS_65, DIMENSIONS_PER_LANDMARK), dtype=np.float32)
        current_idx = 0

        if results.pose_landmarks:
            for landmark_idx_mp in UPPER_POSE_LANDMARK_INDICES:
                if landmark_idx_mp < len(results.pose_landmarks.landmark):
                    landmark = results.pose_landmarks.landmark[landmark_idx_mp]
                    current_frame_keypoints_65[current_idx, :] = [landmark.x, landmark.y, landmark.z]
                current_idx += 1
        else:
            current_idx += NUM_UPPER_POSE_LANDMARKS

        if results.left_hand_landmarks:
            for landmark_idx_hand in range(NUM_HAND_LANDMARKS):
                if landmark_idx_hand < len(results.left_hand_landmarks.landmark):
                    landmark = results.left_hand_landmarks.landmark[landmark_idx_hand]
                    if current_idx < TOTAL_LANDMARKS_65:
                        current_frame_keypoints_65[current_idx, :] = [landmark.x, landmark.y, landmark.z]
                current_idx += 1
        else:
            current_idx += NUM_HAND_LANDMARKS

        if results.right_hand_landmarks:
            for landmark_idx_hand in range(NUM_HAND_LANDMARKS):
                if landmark_idx_hand < len(results.right_hand_landmarks.landmark):
                    landmark = results.right_hand_landmarks.landmark[landmark_idx_hand]
                    if current_idx < TOTAL_LANDMARKS_65:
                        current_frame_keypoints_65[current_idx, :] = [landmark.x, landmark.y, landmark.z]
                current_idx += 1
        else:
            if current_idx < TOTAL_LANDMARKS_65: # Should not be strictly necessary if logic is correct
                current_idx += NUM_HAND_LANDMARKS
        
        all_frames_keypoints_65.append(current_frame_keypoints_65)
    
    holistic_model.close()

    # If fewer frames were processed than target_frames (e.g. frames_list was shorter than target_frames)
    # and we need to pad to ensure the output array has `target_frames` rows.
    # This assumes that if padding is needed, it should be with zero keypoints.
    # However, S01 handles frame content padding; this would be keypoint padding.
    # For consistency, if S01 guarantees `len(frames_list) == target_frames`, this isn't needed.
    # Let's assume `frames_list` already contains `target_frames` frames (possibly padded by S01).
    
    if len(all_frames_keypoints_65) != target_frames:
        print(f"    ERROR: Number of processed keypoint sets ({len(all_frames_keypoints_65)}) doesn't match target_frames ({target_frames}). This may indicate an issue with S01 padding or input frames_list length.")
        # To be robust, you might pad here with zeros if absolutely necessary,
        # but it's better if S01 ensures frames_list has target_frames elements.
        # For now, let's return None if this condition isn't met, highlighting a potential upstream issue.
        return None
        
    keypoints_65_raw_arr = np.array(all_frames_keypoints_65, dtype=np.float32)
    if keypoints_65_raw_arr.shape != expected_shape_65:
        print(f"    ERROR: Shape mismatch for raw 65-point array: {keypoints_65_raw_arr.shape} vs {expected_shape_65}")
        return None
    return keypoints_65_raw_arr

# --- Function that reads frames from disk (original file-based approach) ---
def get_processed_65_keypoints_from_frames_on_disk( # Renamed for clarity
    instance_frames_dir: str,
    target_frames_per_instance: int,
    model_complexity_setting: int = DEFAULT_MODEL_COMPLEXITY,
    save_intermediate_dir: Optional[str] = None
) -> Optional[np.ndarray]:
    print(f"  Processing 65 keypoints for frames in: {instance_frames_dir}")
    if save_intermediate_dir:
        os.makedirs(save_intermediate_dir, exist_ok=True)
        print(f"    Intermediate 65-point arrays will be saved to: {save_intermediate_dir}")

    frames_list = load_and_check_frames(instance_frames_dir, target_frames_per_instance)
    if frames_list is None: return None

    keypoints_65_raw = _extract_raw_65_keypoints_from_frames(frames_list, model_complexity_setting, target_frames_per_instance)
    del frames_list # Free memory from loaded frames
    if keypoints_65_raw is None: return None
    if save_intermediate_dir:
        np.save(Path(save_intermediate_dir) / "keypoints_65_raw.npy", keypoints_65_raw)
        print(f"      Saved: keypoints_65_raw.npy (Shape: {keypoints_65_raw.shape})")

    keypoints_65_interpolated = interpolate_missing_keypoints(keypoints_65_raw)
    if keypoints_65_interpolated is None: return None
    if save_intermediate_dir:
        np.save(Path(save_intermediate_dir) / "keypoints_65_interpolated.npy", keypoints_65_interpolated)
        print(f"      Saved: keypoints_65_interpolated.npy (Shape: {keypoints_65_interpolated.shape})")

    keypoints_65_normalized = normalize_keypoints(
        keypoints_65_interpolated,
        LEFT_SHOULDER_IDX_IN_65,
        RIGHT_SHOULDER_IDX_IN_65,
        NUM_UPPER_POSE_LANDMARKS
    )
    if keypoints_65_normalized is None: return None
    if save_intermediate_dir:
        np.save(Path(save_intermediate_dir) / "keypoints_65_normalized.npy", keypoints_65_normalized)
        print(f"      Saved: keypoints_65_normalized.npy (Shape: {keypoints_65_normalized.shape})")
    
    print(f"  Successfully processed 65 keypoints. Shape: {keypoints_65_normalized.shape}")
    return keypoints_65_normalized

# --- Main function for file-based pipeline ---
def run_keypoint_extraction_27_pipeline(
    instance_frames_dir: str,
    target_frames_per_instance: int,
    model_complexity_setting: int = DEFAULT_MODEL_COMPLEXITY,
    save_intermediate_dir: Optional[str] = None
) -> Optional[np.ndarray]:
    current_expected_final_shape_27 = (target_frames_per_instance, TOTAL_LANDMARKS_TO_EXTRACT, DIMENSIONS_PER_LANDMARK)

    keypoints_65_processed = get_processed_65_keypoints_from_frames_on_disk( # Use renamed function
        instance_frames_dir, target_frames_per_instance, model_complexity_setting, save_intermediate_dir
    )
    if keypoints_65_processed is None: return None

    try:
        keypoints_27_final = keypoints_65_processed[:, SELECTED_INDICES_FROM_65_TO_27, :]
    except IndexError as e:
        print(f"    ERROR: Indexing error during 27-point selection: {e}")
        print(f"    Shape of processed 65 keypoints was: {keypoints_65_processed.shape}")
        print(f"    Max index in selection: {max(SELECTED_INDICES_FROM_65_TO_27) if SELECTED_INDICES_FROM_65_TO_27 else 'N/A'}")
        return None
    
    if keypoints_27_final.shape != current_expected_final_shape_27:
        print(f"    ERROR: Final 27-point array shape mismatch: {keypoints_27_final.shape} vs {current_expected_final_shape_27}")
        return None
    
    return keypoints_27_final


# --- NEW FUNCTION: In-Memory Keypoint Extraction Pipeline ---
def extract_keypoints_from_memory_frames(
    frames_list: List[np.ndarray],
    target_frames_count: int, # This should match the number of frames S01 prepared (incl. padding)
    model_complexity_setting: int = DEFAULT_MODEL_COMPLEXITY
) -> Optional[np.ndarray]:
    """
    Extracts and processes keypoints directly from a list of in-memory frames.
    Returns the final (frames, 27, 3) NumPy array of keypoints.
    """
    print(f"  Starting in-memory keypoint extraction for {len(frames_list)} provided frames (target output frames: {target_frames_count}).")

    if not frames_list:
        print("  In-Memory Error: Empty frames_list provided.")
        return None
    
    if len(frames_list) != target_frames_count:
        print(f"  In-Memory Warning: Input frames_list length ({len(frames_list)}) differs from target_frames_count ({target_frames_count}). Ensure S01 provided correctly padded frames.")
        # If S01 always provides target_frames_count elements, this condition implies an issue.
        # For robustness, we could proceed with len(frames_list) but it might lead to shape errors later
        # if the GCN expects exactly target_frames_count.
        # Let's assume S01 does its job and provides target_frames_count frames. If not, this is an issue.

    # Step 1: Extract raw 65 keypoints using the existing helper
    # _extract_raw_65_keypoints_from_frames expects `target_frames` for its output shape.
    keypoints_65_raw = _extract_raw_65_keypoints_from_frames(
        frames_list, model_complexity_setting, target_frames_count
    )
    # `frames_list` can be deleted here if memory is extremely critical, but `_extract_raw...` doesn't modify it
    # del frames_list # Not strictly necessary as it's passed by value (reference)

    if keypoints_65_raw is None:
        print("  In-Memory Error: Failed to extract raw 65 keypoints.")
        return None
    # print(f"    In-Memory: Raw 65 keypoints shape: {keypoints_65_raw.shape}")

    # Step 2: Interpolate missing keypoints
    keypoints_65_interpolated = interpolate_missing_keypoints(keypoints_65_raw)
    if keypoints_65_interpolated is None:
        print("  In-Memory Error: Failed to interpolate 65 keypoints.")
        return None
    # print(f"    In-Memory: Interpolated 65 keypoints shape: {keypoints_65_interpolated.shape}")

    # Step 3: Normalize keypoints
    keypoints_65_normalized = normalize_keypoints(
        keypoints_65_interpolated,
        LEFT_SHOULDER_IDX_IN_65,
        RIGHT_SHOULDER_IDX_IN_65,
        NUM_UPPER_POSE_LANDMARKS
    )
    if keypoints_65_normalized is None:
        print("  In-Memory Error: Failed to normalize 65 keypoints.")
        return None
    # print(f"    In-Memory: Normalized 65 keypoints shape: {keypoints_65_normalized.shape}")

    # Step 4: Select the final 27 keypoints
    current_expected_final_shape_27 = (target_frames_count, TOTAL_LANDMARKS_TO_EXTRACT, DIMENSIONS_PER_LANDMARK)
    try:
        keypoints_27_final = keypoints_65_normalized[:, SELECTED_INDICES_FROM_65_TO_27, :]
    except IndexError as e:
        print(f"  In-Memory ERROR: Indexing error during 27-point selection: {e}")
        print(f"    Shape of processed 65 keypoints was: {keypoints_65_normalized.shape}")
        return None
    
    if keypoints_27_final.shape != current_expected_final_shape_27:
        print(f"  In-Memory ERROR: Final 27-point array shape mismatch: {keypoints_27_final.shape} vs {current_expected_final_shape_27}")
        return None

    print(f"  Successfully extracted and processed {keypoints_27_final.shape[1]} keypoints for {keypoints_27_final.shape[0]} frames in-memory.")
    return keypoints_27_final


# --- Standalone Execution Block (Example for testing this module) ---
if __name__ == "__main__":
    import argparse
    import shutil
    
    parser = argparse.ArgumentParser(description="Test S02_Keypoint_Extractor.py functions.")
    parser.add_argument("--test_frames_dir", type=str, help="Directory with frames for a single instance to test file-based pipeline.")
    parser.add_argument("--test_output_dir", type=str, default="s02_test_output", help="Directory to save test outputs.")
    parser.add_argument("--test_target_frames", type=int, default=DEFAULT_TARGET_FRAMES, help="Number of frames expected/to process.")
    parser.add_argument("--mp_complexity_test", type=int, default=DEFAULT_MODEL_COMPLEXITY, choices=[0,1,2], help="MP Complexity for test.")
    parser.add_argument("--save_intermediates_test", action=argparse.BooleanOptionalAction, default=True, help="Save intermediate 65-pt arrays during file-based test.")
    
    args = parser.parse_args()

    # --- Test file-based pipeline ---
    if args.test_frames_dir:
        print(f"\n--- Running Standalone Test for S02 (File-Based Pipeline) ---")
        current_script_dir_for_test = Path(__file__).resolve().parent if '__file__' in locals() else Path(".").resolve()
        input_frames_path_for_test_run = Path(args.test_frames_dir)
        if not input_frames_path_for_test_run.is_absolute():
            input_frames_path_for_test_run = current_script_dir_for_test / input_frames_path_for_test_run
        output_base_dir_for_keypoints_resolved_test = Path(args.test_output_dir)
        if not output_base_dir_for_keypoints_resolved_test.is_absolute():
            output_base_dir_for_keypoints_resolved_test = current_script_dir_for_test / output_base_dir_for_keypoints_resolved_test
        
        if not input_frames_path_for_test_run.is_dir():
            print(f"ERROR: Input frames dir '{input_frames_path_for_test_run.resolve()}' not found for file-based test.")
        else:
            instance_name_stem_test = input_frames_path_for_test_run.name
            instance_specific_output_dir_test = output_base_dir_for_keypoints_resolved_test / instance_name_stem_test
            os.makedirs(instance_specific_output_dir_test, exist_ok=True)
            intermediate_save_path_test = None
            if args.save_intermediates_test:
                intermediate_save_path_test = instance_specific_output_dir_test / "intermediate_65pt_stages_filebased"
                os.makedirs(intermediate_save_path_test, exist_ok=True)

            print(f"Testing file-based pipeline for: {input_frames_path_for_test_run.resolve()}")
            final_keypoints_filebased = run_keypoint_extraction_27_pipeline(
                instance_frames_dir=str(input_frames_path_for_test_run),
                target_frames_per_instance=args.test_target_frames,
                model_complexity_setting=args.mp_complexity_test,
                save_intermediate_dir=str(intermediate_save_path_test) if intermediate_save_path_test else None
            )
            if final_keypoints_filebased is not None:
                print(f"  File-based test successful. Final keypoints shape: {final_keypoints_filebased.shape}")
                np.save(instance_specific_output_dir_test / "test_final_27_keypoints_filebased.npy", final_keypoints_filebased)
            else:
                print("  File-based test failed.")
    else:
        print("\nS02_Keypoint_Extractor.py: No --test_frames_dir provided for file-based standalone test.")

    # --- Test in-memory pipeline with dummy frames ---
    print(f"\n--- Running Standalone Test for S02 (In-Memory Pipeline with Dummy Frames) ---")
    dummy_frames_list: List[np.ndarray] = []
    dummy_target_frames_mem_test = 5 # Should match S01's target_frames for this test
    for i in range(dummy_target_frames_mem_test):
        img = np.zeros((100,100,3), dtype=np.uint8)
        cv2.putText(img, f"M{i}", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
        dummy_frames_list.append(img)
    print(f"Created {len(dummy_frames_list)} dummy frames in memory for testing.")

    final_keypoints_inmemory = extract_keypoints_from_memory_frames(
        frames_list=dummy_frames_list,
        target_frames_count=dummy_target_frames_mem_test, # This is the T in (T, V, C)
        model_complexity_setting=0 # Use complexity 0 for faster test
    )
    if final_keypoints_inmemory is not None:
        print(f"  In-memory test successful. Final keypoints shape: {final_keypoints_inmemory.shape}")
        # Optionally save if you want to inspect
        # np.save(Path(args.test_output_dir) / "test_final_27_keypoints_inmemory_dummy.npy", final_keypoints_inmemory)
    else:
        print("  In-memory test failed.")