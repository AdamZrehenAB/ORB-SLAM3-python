import orbslam3
import argparse
from glob import glob
import os 
import cv2
import csv
import numpy as np


def create_annotated_video(vocab_file, settings_file, dataset_path, output_video_path, num_frames=30, save_frames=True, save_trajectory_csv=True):
    """
    Process the first N frames and create an MP4 video from annotated frames.
    
    Args:
        vocab_file: Path to ORB vocabulary file
        settings_file: Path to settings file
        dataset_path: Path to image dataset
        output_video_path: Path for output MP4 file
        num_frames: Number of frames to process (default: 30)
        save_frames: Whether to save individual frames (default: True)
        save_trajectory_csv: Whether to save trajectory points to CSV (default: True)
    """
    
    img_files = sorted(glob(os.path.join(dataset_path, '*.png')))
    print(f'Total images found: {len(img_files)}')
    print(f'Processing first {num_frames} frames...')
    
    if len(img_files) == 0:
        print("No images found in dataset path!")
        return
    
    # Create output directory for frames if saving frames
    frames_output_dir = None
    if save_frames:
        frames_output_dir = output_video_path.replace('.mp4', '_frames')
        os.makedirs(frames_output_dir, exist_ok=True)
        print(f"Saving individual frames to: {frames_output_dir}")
    
    # Initialize SLAM
    slam = orbslam3.system(vocab_file, settings_file, orbslam3.Sensor.MONOCULAR)
    slam.set_use_viewer(False)  # Disable viewer since we're creating video
    slam.initialize()
    
    # Get first image to determine video dimensions
    first_img = cv2.imread(img_files[0])
    if first_img is None:
        print(f"Failed to load first image: {img_files[0]}")
        slam.shutdown()
        return
    
    height, width = first_img.shape[:2]
    print(f"Video dimensions: {width}x{height}")
    
    # Initialize video writer
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 10.0, (width, height))
    except Exception as e:
        print(f"Failed to create video writer: {e}")
        slam.shutdown()
        return
    
    if not video_writer.isOpened():
        print(f"Failed to create video writer for {output_video_path}")
        slam.shutdown()
        return

    frame_count = 0
    processed_frames = 0
    initialization_frames = 0
    trajectory_points = []  # Store trajectory points for visualization
    trajectory_data = []  # Store trajectory data with metadata for CSV
    
    for k, img_file in enumerate(img_files):
        if frame_count >= num_frames:
            break
            
        timestamp = img_file.split('/')[-1][:-4]
        print(f'Processing frame {frame_count + 1}/{num_frames}: {timestamp}')
        
        # Load image
        img = cv2.imread(img_file)
        if img is None:
            print(f"Failed to load image: {img_file}")
            continue
        
        # Process frame
        success = slam.process_image_mono(img, float(timestamp))
        print(f"Success: {success}")
        
        # Get trajectory data (only after SLAM has initialized)
        if k >= 3:
            try:
                trajectory = slam.get_trajectory()
                if len(trajectory) > 0:
                    # Get the latest pose (last element in trajectory)
                    latest_pose = trajectory[-1]
                    # Extract position (translation part of the 4x4 matrix)
                    position = latest_pose[:3, 3]  # x, y, z coordinates
                    trajectory_points.append(position)
                    
                    # Store trajectory data with metadata for CSV
                    trajectory_data.append({
                        'frame_number': frame_count + 1,
                        'timestamp': timestamp,
                        'x_meters': float(position[0]),
                        'y_meters': float(position[1]), 
                        'z_meters': float(position[2]),
                        'success': success
                    })
                    
                    print(f"Trajectory point {len(trajectory_points)}: {position}")
            except Exception as e:
                print(f"Warning: Could not get trajectory data: {e}")
                trajectory = []
        
        # Get annotated frame
        annotated_frame = slam.get_annotated_frame()
        
        if annotated_frame is not None and annotated_frame.size > 0:            
            # Save individual frame if requested
            if save_frames and frames_output_dir:
                frame_filename = f"frame_{frame_count+1:03d}_{timestamp}.png"
                frame_path = os.path.join(frames_output_dir, frame_filename)
                cv2.imwrite(frame_path, annotated_frame)
                print(f"Saved frame: {frame_filename}")
            
            # Frame is already BGR
            if k >= 3:
                # Resize frame to match video writer dimensions
                if annotated_frame.shape[:2] != (height, width):
                    annotated_frame = cv2.resize(annotated_frame, (width, height))
                
                # Add trajectory visualization
                if len(trajectory_points) > 1:
                    try:
                        annotated_frame = draw_trajectory(annotated_frame, trajectory_points)
                    except Exception as e:
                        print(f"Warning: Could not draw trajectory: {e}")
                
                print(f"Writing frame {k+1} to video, shape: {annotated_frame.shape}, dtype: {annotated_frame.dtype}")
                video_writer.write(annotated_frame)
                processed_frames += 1

        else:
            print(f"No annotated frame available for {timestamp}, using original")
            # Use original image if no annotated frame
            if save_frames and frames_output_dir:
                frame_filename = f"frame_{frame_count+1:03d}_{timestamp}_original.png"
                frame_path = os.path.join(frames_output_dir, frame_filename)
                cv2.imwrite(frame_path, img)
                print(f"Saved original frame: {frame_filename}")
            
            video_writer.write(img)
            processed_frames += 1
        
        frame_count += 1
    
    # Save trajectory data to CSV
    if save_trajectory_csv and trajectory_data:
        csv_filename = output_video_path.replace('.mp4', '_trajectory.csv')
        save_trajectory_to_csv(trajectory_data, csv_filename)
        print(f"Trajectory data saved to: {csv_filename}")
    
    # Release resources
    video_writer.release()
    slam.shutdown()
    
    print(f"Video creation complete!")
    print(f"Processed {frame_count} frames")
    print(f"Added {processed_frames} frames to video")
    print(f"Trajectory points collected: {len(trajectory_points)}")
    print(f"Output video: {output_video_path}")
    if save_frames and frames_output_dir:
        print(f"Individual frames saved to: {frames_output_dir}")
    
    # Check if video file was created successfully
    if os.path.exists(output_video_path):
        file_size = os.path.getsize(output_video_path)
        print(f"Video file size: {file_size / (1024*1024):.2f} MB")
    else:
        print("Warning: Video file was not created!")


def save_trajectory_to_csv(trajectory_data, csv_filename):
    """
    Save trajectory data to CSV file.
    
    Args:
        trajectory_data: List of dictionaries containing trajectory information
        csv_filename: Output CSV file path
    """
    if not trajectory_data:
        print("No trajectory data to save")
        return
    
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['frame_number', 'timestamp', 'x_meters', 'y_meters', 'z_meters', 'success']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header
            writer.writeheader()
            
            # Write trajectory data
            for data in trajectory_data:
                writer.writerow(data)
        
        print(f"Trajectory CSV saved successfully: {csv_filename}")
        print(f"Total trajectory points: {len(trajectory_data)}")
        
        # Print some statistics
        if len(trajectory_data) > 1:
            x_coords = [d['x_meters'] for d in trajectory_data]
            y_coords = [d['y_meters'] for d in trajectory_data]
            z_coords = [d['z_meters'] for d in trajectory_data]
            
            print(f"Trajectory statistics:")
            print(f"  X range: {min(x_coords):.3f} to {max(x_coords):.3f} meters")
            print(f"  Y range: {min(y_coords):.3f} to {max(y_coords):.3f} meters")
            print(f"  Z range: {min(z_coords):.3f} to {max(z_coords):.3f} meters")
            
            # Calculate total distance traveled
            total_distance = 0
            for i in range(1, len(trajectory_data)):
                prev_pos = np.array([trajectory_data[i-1]['x_meters'], 
                                   trajectory_data[i-1]['y_meters'], 
                                   trajectory_data[i-1]['z_meters']])
                curr_pos = np.array([trajectory_data[i]['x_meters'], 
                                   trajectory_data[i]['y_meters'], 
                                   trajectory_data[i]['z_meters']])
                total_distance += np.linalg.norm(curr_pos - prev_pos)
            
            print(f"  Total distance traveled: {total_distance:.3f} meters")
            
    except Exception as e:
        print(f"Error saving trajectory CSV: {e}")


def draw_trajectory(frame, trajectory_points):
    """
    Draw trajectory on the frame.
    
    Args:
        frame: Input frame
        trajectory_points: List of 3D positions (x, y, z)
    
    Returns:
        Frame with trajectory drawn
    """
    if len(trajectory_points) < 2:
        return frame
    
    # Create a copy of the frame
    result = frame.copy()
    
    # Convert 3D trajectory to 2D for visualization
    # We'll project the trajectory onto a 2D plane
    trajectory_2d = []
    
    # Scale factors for visualization
    scale_x = frame.shape[1] / 20.0  # Map 20 meters to frame width
    scale_y = frame.shape[0] / 20.0  # Map 20 meters to frame height
    
    # Center offset
    center_x = frame.shape[1] // 2
    center_y = frame.shape[0] // 2
    
    for point in trajectory_points:
        # Project 3D point to 2D (simple top-down view)
        x_2d = int(center_x + point[0] * scale_x)  # X coordinate
        y_2d = int(center_y - point[2] * scale_y)  # Z coordinate (negative for correct orientation)
        
        # Clamp to frame bounds
        x_2d = max(0, min(x_2d, frame.shape[1] - 1))
        y_2d = max(0, min(y_2d, frame.shape[0] - 1))
        
        trajectory_2d.append((x_2d, y_2d))
    
    # Draw trajectory line
    if len(trajectory_2d) > 1:
        for i in range(1, len(trajectory_2d)):
            cv2.line(result, trajectory_2d[i-1], trajectory_2d[i], (0, 255, 0), 2)
    
    # Draw trajectory points
    for i, point in enumerate(trajectory_2d):
        color = (0, 0, 255) if i == len(trajectory_2d) - 1 else (255, 0, 0)  # Red for current, blue for others
        cv2.circle(result, point, 5, color, -1)
    
    # Add trajectory info text
    cv2.putText(result, f"Trajectory: {len(trajectory_points)} points", 
               (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create annotated video from ORB-SLAM3")
    parser.add_argument("--vocab_file", default="third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt", 
                       help="Path to ORB vocabulary file")
    parser.add_argument("--settings_file", default="third_party/ORB_SLAM3/Examples/Monocular/TUM1_corrected.yaml", 
                       help="Path to settings file")
    parser.add_argument("--dataset_path", 
                       default="/home/adamz/git/liquid/flow_fs/data_old/VH_data_Sheet1/20250325062137/20250325062137-HU-front_center/frames", 
                       help="Path to image dataset")
    parser.add_argument("--output_video", default="annotated_slam_video.mp4", 
                       help="Output video file path")
    parser.add_argument("--num_frames", type=int, default=500,
                       help="Number of frames to process (default: 50)")
    parser.add_argument("--no_save_frames", action="store_true", 
                       help="Don't save individual frames")
    parser.add_argument("--no_save_trajectory_csv", action="store_true", 
                       help="Don't save trajectory data to CSV")
    
    args = parser.parse_args()
    
    create_annotated_video(
        vocab_file=args.vocab_file,
        settings_file=args.settings_file,
        dataset_path=args.dataset_path,
        output_video_path=args.output_video,
        num_frames=args.num_frames,
        save_frames=not args.no_save_frames,
        save_trajectory_csv=not args.no_save_trajectory_csv
    ) 