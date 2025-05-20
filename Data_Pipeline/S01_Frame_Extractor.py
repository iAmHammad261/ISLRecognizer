import os
import re
# from tqdm import tqdm # tqdm is used in the training script's main loop, not directly in this func
import cv2
import numpy as np
import math
from pathlib import Path
from typing import List, Optional # Added for type hinting

def get_frame_filename_pattern(target_frames: int) -> str:
    """
    Calculates the filename pattern based on the number of target frames.
    """
    if target_frames <= 0:
        raise ValueError("target_frames must be a positive integer.")
    padding_width = len(str(target_frames - 1)) if target_frames > 1 else 1
    return f'frame_{{:0{padding_width}d}}.png'

def extract_and_save_frames(
    video_path: str,
    output_dir_for_frames: str,
    target_frames: int
) -> bool:
    """
    Extracts exactly target_frames from a single video using uniform sampling,
    saves them to the specified output directory.
    This version aligns its padding logic with the user's training script:
    if reads fail, padding with the last good frame happens by appending to the end.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return False

    try:
        os.makedirs(output_dir_for_frames, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory {output_dir_for_frames}: {e}")
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False

    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames_in_video <= 0:
        print(f"Warning: Video {video_path} has 0 frames or metadata error.")
        cap.release()
        return False

    successfully_read_frames_list = []
    last_successful_frame = None

    sampling_points = np.linspace(0, total_frames_in_video, target_frames, endpoint=False)
    indices_to_read = np.floor(sampling_points).astype(int)
    indices_to_read = np.maximum(0, np.minimum(indices_to_read, total_frames_in_video - 1))

    for i, frame_idx in enumerate(indices_to_read):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            successfully_read_frames_list.append(frame)
            last_successful_frame = frame
        else:
            print(f"Warning: Could not read frame at original index {frame_idx} (target slot {i+1}/{target_frames}) from {video_path}.")

    cap.release()

    final_output_frames = list(successfully_read_frames_list)

    if len(final_output_frames) < target_frames:
        missing_count = target_frames - len(final_output_frames)
        print(f"Warning: Only read {len(final_output_frames)}/{target_frames} frames successfully from {video_path}.")

        if last_successful_frame is not None:
            print(f"Padding with {missing_count} copies of the last successfully read frame.")
            for _ in range(missing_count):
                final_output_frames.append(last_successful_frame.copy())
        else:
            print(f"Error: No frames were successfully read from {video_path}. Cannot pad or save.")
            return False

    if len(final_output_frames) != target_frames:
        print(f"Error: Frame processing resulted in an incorrect number of frames ({len(final_output_frames)} vs {target_frames}) for {video_path} even after padding attempts.")
        return False

    frame_filename_pattern = get_frame_filename_pattern(target_frames)
    try:
        for i, frame_to_save in enumerate(final_output_frames):
            frame_filename = frame_filename_pattern.format(i)
            frame_filepath = os.path.join(output_dir_for_frames, frame_filename)
            # Save with no compression for PNG to be faster, though PNG is generally lossless.
            # If speed is absolutely critical and lossy is acceptable, JPEG could be considered,
            # but PNG is common for frame extraction for datasets.
            cv2.imwrite(frame_filepath, frame_to_save, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"Successfully extracted and saved {target_frames} frames to {output_dir_for_frames}")
        return True
    except Exception as e:
        print(f"Error saving frames to {output_dir_for_frames}: {e}")
        return False

def extract_frames_to_memory(
    video_path: str,
    target_frames: int
) -> Optional[List[np.ndarray]]:
    """
    Extracts exactly target_frames from a single video using uniform sampling,
    returns them as a list of NumPy arrays (frames).
    Padding with the last successfully read frame is applied if needed.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames_in_video <= 0:
        print(f"Warning: Video {video_path} has 0 frames or metadata error.")
        cap.release()
        return None

    # This list will store only successfully read frames initially
    successfully_read_frames_list: List[np.ndarray] = []
    last_successful_frame: Optional[np.ndarray] = None

    # Uniform sampling logic
    sampling_points = np.linspace(0, total_frames_in_video, target_frames, endpoint=False)
    indices_to_read = np.floor(sampling_points).astype(int)
    # Ensure indices are within the valid range [0, total_frames_in_video - 1]
    indices_to_read = np.maximum(0, np.minimum(indices_to_read, total_frames_in_video - 1))

    # Read selected frames
    for i, frame_idx in enumerate(indices_to_read):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            successfully_read_frames_list.append(frame)
            last_successful_frame = frame # Keep track of the latest good frame
        else:
            print(f"Warning: Could not read frame at original index {frame_idx} (target slot {i+1}/{target_frames}) from {video_path}.")
            # Shortfall will be handled by padding after the loop.

    cap.release()

    # Now, handle padding similar to the training script's logic
    final_output_frames: List[np.ndarray] = list(successfully_read_frames_list) # Start with what was successfully read

    if len(final_output_frames) < target_frames:
        missing_count = target_frames - len(final_output_frames)
        print(f"Warning: Only read {len(final_output_frames)}/{target_frames} frames successfully from {video_path}.")

        if last_successful_frame is not None: # Check if there's any frame to pad with
            print(f"Padding with {missing_count} copies of the last successfully read frame.")
            for _ in range(missing_count):
                final_output_frames.append(last_successful_frame.copy()) # Append copies to the end
        else:
            # This means no frames were read successfully at all.
            print(f"Error: No frames were successfully read from {video_path}. Cannot pad or provide frames.")
            return None # Return None as we cannot fulfill the request

    # After potential padding, check if we have the target number of frames
    if len(final_output_frames) != target_frames:
        print(f"Error: Frame processing resulted in an incorrect number of frames ({len(final_output_frames)} vs {target_frames}) for {video_path} even after padding attempts.")
        return None # Return None as the condition is not met

    print(f"Successfully extracted {len(final_output_frames)} frames into memory from {video_path}")
    return final_output_frames


# --- Example of how this script could be tested standalone (optional) ---
if __name__ == "__main__":
    import shutil
    print("S01_Frame_Extractor.py - Aligned with training script's padding logic.")
    print("This block is for testing functions directly.")

    # --- Test parameters ---
    test_video_filename = "dummy_test_video_for_S01.mp4"
    test_output_dir_for_save = "test_S01_extracted_frames_save" # For extract_and_save_frames
    test_target_frames = 10
    actual_frames_in_dummy_video = 5 # To test padding

    # --- Create a dummy video ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use 'mp4v' for .mp4
    out_dummy = cv2.VideoWriter(test_video_filename, fourcc, 20.0, (100, 100))
    for i in range(actual_frames_in_dummy_video):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(img, str(i), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out_dummy.write(img)
    out_dummy.release()
    print(f"Created dummy video: {test_video_filename} with {actual_frames_in_dummy_video} frames.")

    if os.path.exists(test_video_filename):
        # --- Test extract_and_save_frames ---
        print(f"\n--- Testing extract_and_save_frames ---")
        print(f"Video: {test_video_filename}, Target frames: {test_target_frames}, Output dir: {test_output_dir_for_save}")
        if os.path.exists(test_output_dir_for_save):
            shutil.rmtree(test_output_dir_for_save)
            print(f"Cleaned up previous test directory: {test_output_dir_for_save}")

        success_save = extract_and_save_frames(test_video_filename, test_output_dir_for_save, test_target_frames)
        if success_save:
            print(f"Test extract_and_save_frames successful. Frames saved in {test_output_dir_for_save}")
            num_files = len([name for name in os.listdir(test_output_dir_for_save) if os.path.isfile(os.path.join(test_output_dir_for_save, name))])
            if num_files == test_target_frames:
                print(f"Correct number of frames ({num_files}) found in output directory.")
            else:
                print(f"ERROR: Incorrect number of frames in output. Found {num_files}, expected {test_target_frames}")
        else:
            print("Test extract_and_save_frames failed.")

        # --- Test extract_frames_to_memory ---
        print(f"\n--- Testing extract_frames_to_memory ---")
        print(f"Video: {test_video_filename}, Target frames: {test_target_frames}")
        
        frames_in_memory = extract_frames_to_memory(test_video_filename, test_target_frames)
        if frames_in_memory is not None:
            print(f"Test extract_frames_to_memory successful. Extracted {len(frames_in_memory)} frames into memory.")
            if len(frames_in_memory) == test_target_frames:
                print(f"Correct number of frames ({len(frames_in_memory)}) returned.")
                # Optionally, display one of the frames if you have a display environment
                # cv2.imshow("First in-memory frame", frames_in_memory[0])
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            else:
                print(f"ERROR: Incorrect number of frames returned. Got {len(frames_in_memory)}, expected {test_target_frames}")
        else:
            print("Test extract_frames_to_memory failed (returned None).")

        # --- Cleanup dummy video ---
        if os.path.exists(test_video_filename):
            os.remove(test_video_filename)
            print(f"\nCleaned up dummy video: {test_video_filename}")
    else:
        print(f"Error: Could not create dummy video for testing at {test_video_filename}")