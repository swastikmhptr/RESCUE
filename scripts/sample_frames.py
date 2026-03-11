import cv2
import os


def sample_frames(video_path, output_folder, sampling_fps=1.0):
    """
    Samples frames from a video at a specified sampling_fps and saves to output_folder.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    # Capture original video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: FPS is 0, cannot sample.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(
        f"Video Properties: Duration={duration:.2f}s, FPS={fps:.2f}, Total Frames={total_frames}"
    )
    print(f"Sampling at {sampling_fps} FPS (one frame every {1 / sampling_fps:.2f}s)")

    frame_count = 0
    saved_count = 0

    # Iterate through the video duration based on sampling_fps
    while cap.isOpened():
        # Calculate the frame index for the current sample
        # saved_count / sampling_fps gives the timestamp in seconds
        # multiplying by video fps gives the frame index
        frame_id = int((saved_count / sampling_fps) * fps)

        if frame_id >= total_frames:
            break

        # Set video position to specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()

        if not ret:
            break

        # Save current frame as image
        output_filename = f"frame_{saved_count:04d}.jpg"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, frame)

        print(
            f"Saved {output_filename} at {saved_count / sampling_fps:.2f}s (Frame {frame_id})"
        )
        saved_count += 1

    cap.release()
    print(f"Finished sampling. Total frames saved: {saved_count}")


if __name__ == "__main__":
    video_path = "generated/disaster city.mp4"
    output_path = "generated/sampled_disaster_city"
    sampling_fps = 3  # Sample at 0.5 fps (one frame every 2 seconds)

    sample_frames(video_path, output_path, sampling_fps=sampling_fps)
