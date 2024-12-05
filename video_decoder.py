# coded in part with https://claude.ai/chat/2fba7438-0e73-41d5-bd98-313e5d0a57cc

import subprocess
import numpy as np
import json
import requests
from thumbnail_api import Rectangle

def decode_video_frames(video_url, start_frame=None, n_frames=None, start_time=None, end_time=None):
    """
    Decode a range of frames from an MP4 video file accessed via HTTPS.
    
    Args:
        video_url (str): HTTPS URL to the MP4 file
        start_frame (int, optional): Starting frame number (0-based)
        n_frames (int, optional): Number of frames to decode
        start_time (float, optional): Starting time in seconds
        end_time (float, optional): Ending time in seconds
    
    Returns:
        tuple: (frames, metadata) where:
            - frames: numpy.ndarray of shape (n_frames, height, width, 3) with RGB values
            - metadata: dict containing the full ffprobe output
        
    Raises:
        RuntimeError: If FFmpeg encounters an error or output size is incorrect
        ValueError: If invalid frame/time parameters are provided
        KeyError: If video stream information is missing expected fields
    """
    # Input validation
    if (start_frame is not None) != (n_frames is not None):
        raise ValueError("Both start_frame and n_frames must be provided together")
    if (start_time is not None) != (end_time is not None):
        raise ValueError("Both start_time and end_time must be provided together")
    if start_frame is not None and start_time is not None:
        raise ValueError("Cannot specify both frame numbers and timestamps")
    
    # Get video information using ffprobe
    probe_cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        '-show_format',
        '-select_streams', 'v:0',
        video_url
    ]
    
    try:
        probe_output, probe_error = subprocess.Popen(
            probe_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        ).communicate()
        metadata = json.loads(probe_output)
        
        if not metadata.get('streams'):
            raise ValueError("No streams found in video file")
            
        # Get the first video stream
        video_stream = metadata['streams'][0]
        
        # Extract video properties
        try:
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            
            # Parse frame rate which might be in different formats
            if 'r_frame_rate' in video_stream:
                num, den = map(int, video_stream['r_frame_rate'].split('/'))
                fps = num / den
            elif 'avg_frame_rate' in video_stream:
                num, den = map(int, video_stream['avg_frame_rate'].split('/'))
                fps = num / den
            else:
                raise KeyError("Could not find frame rate information")
                
        except KeyError as e:
            raise KeyError(f"Missing required video property: {str(e)}")
            
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFprobe error: {e.stderr.decode()}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse FFprobe output: {str(e)}")
    
    # Calculate duration based on input parameters
    if start_frame is not None and n_frames is not None:
        start_time = start_frame / fps
        duration = n_frames / fps
        expected_frames = n_frames
    elif start_time is not None and end_time is not None:
        duration = end_time - start_time
        expected_frames = int(duration * fps)
    else:
        raise ValueError("Either frame numbers or timestamps must be provided")
    
    # Build ffmpeg command
    cmd = ['ffmpeg', '-ss', str(start_time)]
    
    # Add time selection
    cmd.extend(['-i', video_url, '-t', str(duration)])
    
    # Add output format settings
    cmd.extend([
        '-f', 'image2pipe',
        '-pix_fmt', 'rgb24',
        '-vcodec', 'rawvideo',
        '-'
    ])
    
    # Run ffmpeg process with communicate()
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8  # Use large buffer size for video data
        )
        raw_data, stderr = process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {stderr.decode()}")
            
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg error: {e.stderr.decode()}")
    
    # Verify the output size
    expected_bytes = width * height * 3 * expected_frames
    actual_bytes = len(raw_data)
    
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"FFmpeg output size mismatch: expected {expected_bytes} bytes "
            f"({expected_frames} frames) but got {actual_bytes} bytes "
            f"({actual_bytes // (width * height * 3)} frames)"
        )
    
    # Reshape into frames
    frames = np.frombuffer(raw_data, dtype=np.uint8)
    frames = frames.reshape((expected_frames, height, width, 3))
    
    return frames, metadata

def download_video(self, start_frame_no: int, nframes: int, rect: Rectangle, subsample: int = 1):
    """
    Download and assemble video tiles into a single numpy array.
    
    Args:
        start_frame_no: Starting frame number
        nframes: Number of frames to download
        rect: Rectangle coordinates after subsampling
        subsample: Subsample factor
    
    Returns:
        numpy.ndarray: Array of shape (nframes, height, width, 3) containing the video data
    """
    level = self.level_from_subsample(subsample)
    level_width = self.width(subsample)
    level_height = self.height(subsample)
    
    # Create output array to hold the final video
    result = np.zeros((nframes, rect.height, rect.width, 3), dtype=np.uint8)
    
    # Compute the tiles that intersect the rectangle
    min_tile_y = rect.y1 // self.tile_height()
    max_tile_y = 1 + (rect.y2 - 1) // self.tile_height()
    min_tile_x = rect.x1 // self.tile_width()
    max_tile_x = 1 + (rect.x2 - 1) // self.tile_width()
    
    for tile_y in range(min_tile_y, max_tile_y):
        for tile_x in range(min_tile_x, max_tile_x):
            tile_url = self.tile_url(level, tile_x, tile_y)
            
            # Check if tile exists
            response = requests.head(tile_url)
            if response.status_code == 404:
                print(f"Warning: tile {tile_x},{tile_y} does not exist, skipping")
                continue
                
            # Calculate tile and intersection rectangles
            tile_rectangle = Rectangle(
                tile_x * self.tile_width(),
                tile_y * self.tile_height(),
                (tile_x + 1) * self.tile_width(),
                (tile_y + 1) * self.tile_height()
            )
            
            intersection = rect.intersection(tile_rectangle)
            if intersection is None:
                continue
                
            # Calculate source and destination rectangles
            src_rect = intersection.translate(-tile_rectangle.x1, -tile_rectangle.y1)
            dest_rect = intersection.translate(-rect.x1, -rect.y1)
            
            print(f"Fetching {tile_url}")
            print(f"From tile {tile_url}, copying {src_rect} to destination {dest_rect}")
            
            try:
                # Download the tile video
                frames, metadata = decode_video_frames(
                    video_url=tile_url,
                    start_frame=start_frame_no,
                    n_frames=nframes
                )
                
                # Copy the intersection region to the result array
                result[:, 
                       dest_rect.y1:dest_rect.y2,
                       dest_rect.x1:dest_rect.x2,
                       :] = frames[:, 
                                 src_rect.y1:src_rect.y2,
                                 src_rect.x1:src_rect.x2,
                                 :]
                
            except Exception as e:
                print(f"Error processing tile {tile_url}: {str(e)}")
                continue
    
    return result
