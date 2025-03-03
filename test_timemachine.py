import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from timemachine import TimeMachine
from thumbnail_api import Rectangle
from video_decoder import decode_video_frames
import dateutil.parser
import pytz
import cv2  # for saving images

def save_frame_as_jpg(frame, filename):
    """Save a single frame as JPG."""
    # Convert from RGB to BGR for cv2
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, frame_bgr)
    print(f"Saved {filename}")

def test_timemachine_download():
    """Test TimeMachine video download against direct tile download."""
    
    # Initialize TimeMachine
    print("Initializing TimeMachine...")
    timemachine = TimeMachine("https://tiles.cmucreatelab.org/ecam/timemachines/clairton4/2024-10-23.timemachine")
    
    # Define the centered rectangle with y-offset of 800
    rect = Rectangle(x1=356, y1=1000, x2=1068, y2=1400)  # y values increased by 800
    
    # Download via TimeMachine
    print("\nDownloading via TimeMachine.download_video...")
    tm_frames = timemachine.download_video_frame_range(
        start_frame_no=0,
        nframes=15,
        rect=rect,
        subsample=2
    )
    
    # Download the tile directly for comparison
    print("\nDownloading tile directly for comparison...")
    tile_url = "https://tiles.cmucreatelab.org/ecam/timemachines/clairton4/2024-10-23.timemachine/crf26-12fps-1424x800/2/4/0.mp4"
    full_frames = decode_video_frames(
        video_url=tile_url,
        start_frame=0,
        n_frames=15
    )
    
    # Crop the directly downloaded frames
    crop_rect = Rectangle(x1=356, y1=200, x2=1068, y2=600)  # Original centered rectangle
    cropped_frames = full_frames[:, crop_rect.y1:crop_rect.y2, crop_rect.x1:crop_rect.x2, :]
    
    # Verify shapes
    print("\nVerifying dimensions...")
    expected_shape = (15, 400, 712, 3)  # 15 frames, half height/width, RGB
    assert tm_frames.shape == expected_shape, f"TimeMachine frames shape {tm_frames.shape} != expected {expected_shape}"
    assert cropped_frames.shape == expected_shape, f"Cropped frames shape {cropped_frames.shape} != expected {expected_shape}"
    
    # Compare the frames
    print("Comparing frame content...")
    assert_array_equal(
        tm_frames,
        cropped_frames,
        err_msg="Frame content mismatch between TimeMachine and direct download"
    )
    
    # Print some statistics
    print("\nVideo Statistics:")
    print(f"Frame dimensions: {tm_frames.shape[1]}x{tm_frames.shape[2]}")
    for channel, color in enumerate(['Red', 'Green', 'Blue']):
        channel_sum = tm_frames[:, :, :, channel].sum()
        print(f"{color} channel sum: {channel_sum:,}")
    
    print("\nFirst test passed successfully!")

def average_pool_2x2(frames):
    """
    Downscale video frames by averaging 2x2 pixel blocks.
    
    Args:
        frames: numpy array of shape (n_frames, height, width, 3)
    
    Returns:
        numpy array of shape (n_frames, height//2, width//2, 3) with dtype uint8
    """
    n_frames, height, width, channels = frames.shape
    pooled = frames.reshape(n_frames, height//2, 2, width//2, 2, channels)
    # Convert to float for averaging, then back to uint8
    return np.round(pooled.mean(axis=(2, 4))).astype(np.uint8)

def test_timemachine_scale():
    """Test TimeMachine video download at different scales."""
    
    print("\n=== Starting scale comparison test ===")
    
    print("Initializing TimeMachine...")
    timemachine = TimeMachine("https://tiles.cmucreatelab.org/ecam/timemachines/clairton4/2024-10-23.timemachine")
    
    # Original rectangle with y-offset of 800
    rect_subsampled = Rectangle(x1=356, y1=1000, x2=1068, y2=1400)
    
    # Download at subsample=2
    print("\nDownloading at subsample=2...")
    frames_subsampled = timemachine.download_video_frame_range(
        start_frame_no=0,
        nframes=15,
        rect=rect_subsampled,
        subsample=2
    )
    
    # Double all coordinates for full resolution
    rect_full = Rectangle(
        x1=rect_subsampled.x1 * 2,
        y1=rect_subsampled.y1 * 2,
        x2=rect_subsampled.x2 * 2,
        y2=rect_subsampled.y2 * 2
    )
    
    print("\nDownloading at full resolution (subsample=1)...")
    frames_full = timemachine.download_video_frame_range(
        start_frame_no=0,
        nframes=15,
        rect=rect_full,
        subsample=1
    )
    
    # Downscale the full resolution frames
    print("Downscaling full resolution frames...")
    frames_downscaled = average_pool_2x2(frames_full)
    
    # Save first frames as JPG for visual comparison
    save_frame_as_jpg(frames_subsampled[0], "subsample2_frame0.jpg")
    save_frame_as_jpg(frames_downscaled[0], "subsample1_downscaled_frame0.jpg")
    
    # Verify shapes
    print("\nVerifying dimensions...")
    assert frames_subsampled.shape == frames_downscaled.shape, (
        f"Shape mismatch: subsampled {frames_subsampled.shape} != "
        f"downscaled {frames_downscaled.shape}"
    )
    
    # Calculate differences and statistics
    diff = frames_subsampled.astype(np.float32) - frames_downscaled.astype(np.float32)
    squared_diff = diff * diff
    rms = np.sqrt(np.mean(squared_diff))
    
    # Calculate mean intensity of both images
    mean_intensity_subsampled = np.mean(frames_subsampled)
    mean_intensity_downscaled = np.mean(frames_downscaled)
    overall_mean = (mean_intensity_subsampled + mean_intensity_downscaled) / 2
    
    # Calculate RMS to mean ratio
    rms_to_mean_ratio = rms / overall_mean
    
    print("\nError Metrics:")
    print(f"RMS error: {rms:.2f}")
    print(f"Mean intensity (subsample=2): {mean_intensity_subsampled:.2f}")
    print(f"Mean intensity (subsample=1 downscaled): {mean_intensity_downscaled:.2f}")
    print(f"Overall mean intensity: {overall_mean:.2f}")
    print(f"RMS/Mean ratio: {rms_to_mean_ratio:.4f}")
    
    # Channel-specific differences
    for channel, color in enumerate(['Red', 'Green', 'Blue']):
        channel_diff = diff[:, :, :, channel]
        channel_rms = np.sqrt(np.mean(channel_diff * channel_diff))
        channel_mean = (np.mean(frames_subsampled[:, :, :, channel]) + 
                       np.mean(frames_downscaled[:, :, :, channel])) / 2
        print(f"\n{color} channel:")
        print(f"  RMS error: {channel_rms:.2f}")
        print(f"  Mean intensity: {channel_mean:.2f}")
        print(f"  RMS/Mean ratio: {(channel_rms/channel_mean):.4f}")
    
    # Save difference visualization
    diff_visualization = np.clip(np.abs(diff[0]) * 10, 0, 255).astype(np.uint8)  # Scale up differences for visibility
    save_frame_as_jpg(diff_visualization, "frame0_differences.jpg")
    
    # Assert based on RMS/Mean ratio instead of maximum difference
    assert rms_to_mean_ratio <= 0.1, f"RMS/Mean ratio {rms_to_mean_ratio:.4f} exceeds threshold of 0.1"
    
    print("\nSecond test passed successfully!")

if __name__ == '__main__':
    # Run both tests
    print("=== Running direct download comparison test ===")
    test_timemachine_download()
    
    print("\n=== Running scale comparison test ===")
    test_timemachine_scale()

def parse_et(date_str):
    """Parse a date string in Eastern Time."""
    eastern_tz = pytz.timezone("America/New_York")
    return eastern_tz.localize(dateutil.parser.parse(date_str))

def check_frameno_from_date(tm, dt):
    be = tm.frameno_from_date_before_or_equal(dt)
    assert tm.capture_datetimes()[be] <= dt
    if be < len(tm.capture_datetimes())-1:
        assert tm.capture_datetimes()[be+1] > dt
    af = tm.frameno_from_date_after_or_equal(dt)
    assert tm.capture_datetimes()[af] >= dt
    if af > 0:
        assert tm.capture_datetimes()[af-1] < dt

def test_timemachine_capture_datetimes():
    """Test TimeMachine capture times."""
    
    # Initialize TimeMachine
    print("Initializing TimeMachine...")
    timemachine = TimeMachine("https://tiles.cmucreatelab.org/ecam/timemachines/clairton4/2024-10-23.timemachine")
    
    # Get capture times
    capture_times = timemachine.capture_datetimes()

    # Verify the number of capture times
    assert len(capture_times) == 28775

    first_datetime = capture_times[0]

    expected_first_datetime = parse_et("2024-10-23 00:00:00")
    # Get the last entry of a the Pandas series
    last_datetime = capture_times[-1]
    expected_last_datetime = parse_et("2024-10-23 23:59:57")
    assert first_datetime == expected_first_datetime
    assert last_datetime == expected_last_datetime

    assert timemachine.frameno_from_date_before_or_equal(expected_first_datetime) == 0
    assert timemachine.frameno_from_date_before_or_equal(expected_last_datetime) == len(capture_times)-1

    check_frameno_from_date(timemachine, parse_et("2024-10-23 03:00:00"))
    check_frameno_from_date(timemachine, parse_et("2024-10-23 03:00:01"))
    check_frameno_from_date(timemachine, parse_et("2024-10-23 03:00:02"))
    check_frameno_from_date(timemachine, parse_et("2024-10-23 03:00:03"))
    check_frameno_from_date(timemachine, parse_et("2024-10-23 12:00:00"))
    check_frameno_from_date(timemachine, parse_et("2024-10-23 12:00:01"))
    check_frameno_from_date(timemachine, parse_et("2024-10-23 12:00:02"))
    check_frameno_from_date(timemachine, parse_et("2024-10-23 12:00:03"))

    

