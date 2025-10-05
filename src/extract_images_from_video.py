import cv2
import os

def extract_frames(video_path, output_dir, step=10, prefix="frame"):
    """
    Extract every n-th frame from a video and save as images.

    Parameters:
    -----------
    video_path : str
        Path to input video file
    output_dir : str
        Directory where extracted frames will be saved
    step : int
        Save every n-th frame (default: 10)
    prefix : str
        Prefix for saved frame filenames
    """

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        return

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        if frame_idx % step == 0:
            filename = os.path.join(output_dir, f"{prefix}_{saved_idx:05d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"✅ Saved {filename}")
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print("🎉 Done!")


if __name__ == "__main__":
    # Example usage
    video_file = '/datadisk/data/agh_projects/camera_calibration_images/20251003_093124.mp4'
    output_folder = '/datadisk/data/agh_projects/camera_calibration_images/20251003_SamsungGalaxyVidoeHorizontal/'        # directory for images
    n = 15                          # extract every 30th frame

    extract_frames(video_file, output_folder, step=n)
