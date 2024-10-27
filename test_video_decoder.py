import pytest
import numpy as np
from numpy.testing import assert_array_equal
from video_decoder import decode_video_frames

def test_video_decoder():
    """Test the video decoder function with combined dimension, content, and frame consistency checks."""
    url = "https://tiles.cmucreatelab.org/ecam/timemachines/clairton4/2024-10-23.timemachine/crf26-12fps-1424x800/3/8/0.mp4"
    
    # Test 1: Basic functionality and dimensions
    print("Fetching first 30 frames...")
    frames_30, metadata = decode_video_frames(
        video_url=url,
        start_frame=0,
        n_frames=30
    )
    
    # Test basic properties
    assert frames_30.shape == (30, 800, 1424, 3), f"Unexpected shape: {frames_30.shape}"
    
    # Test that no color channel is completely empty
    for channel, color in enumerate(['Red', 'Green', 'Blue']):
        channel_sum = frames_30[:, :, :, channel].sum()
        assert channel_sum > 0, f"Color channel {color} appears to be empty (sum = 0)"
        print(f"{color} channel sum: {channel_sum:,}")
    
    print("First 30 frames passed dimension and content checks")
    
    # Test 2: Frame consistency with subset
    print("\nFetching frames 5-25...")
    frames_subset, _ = decode_video_frames(
        video_url=url,
        start_frame=5,
        n_frames=20
    )
    
    # Verify dimensions of subset
    assert frames_subset.shape == (20, 800, 1424, 3), f"Unexpected shape for subset: {frames_subset.shape}"
    
    # Compare frames
    print("Comparing frames...")
    matching_frames = frames_30[5:25]  # Frames 5-24 from first fetch
    assert_array_equal(
        frames_subset,
        matching_frames,
        err_msg="Frame mismatch between fetches"
    )
    
    print("Frame consistency check passed")
    
    # Print video metadata
    print("\nVideo Metadata:")
    stream = metadata['streams'][0]
    print(f"Codec: {stream.get('codec_name')}")
    print(f"Frame rate: {stream.get('r_frame_rate')}")
    print(f"Duration: {metadata['format'].get('duration')}s")
    

if __name__ == '__main__':
    test_video_decoder()