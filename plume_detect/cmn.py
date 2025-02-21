import datetime
import json
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import subprocess

from typing import Generic, TypeVar

TIME_MACHINES = "https://tiles.cmucreatelab.org/ecam/timemachines"

CAMERAS = {
    "Clairton Coke Works": "clairton4",
    "Shell Plastics West": "vanport3",
    "Edgar Thomson South": "westmifflin2",
    "Metalico": "accan2",
    "Revolution ETC/Harmon Creek Gas Processing Plants": "cryocam",
    "Riverside Concrete": "cementcam",
    "Shell Plastics East": "center1",
    "Irvin": "irvin1",
    "North Shore": "heinz",
    "Mon. Valley": "walnuttowers1",
    "Downtown": "trimont1",
    "Oakland": "oakland"
}


def decode_video_frames(video_url, start_frame=None, n_frames=None, start_time=None, end_time=None):
    # Input validation
    if (start_frame is not None) ^ ((n_frames is not None) or (end_time is not None)):
        raise ValueError("Both start_frame and n_frames must be provided together")

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
    cmd = ['ffmpeg', '-ss', str(start_time), '-t', str(duration)]

    # Add video url
    cmd.extend(['-i', video_url])

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
            bufsize=10 ** 8  # Use large buffer size for video data
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


def get_camera_id(name: str):
    if name in CAMERAS:
        return CAMERAS[name]

    for cam in CAMERAS.values():
        if cam == name:
            return cam

    return None


def get_time(t: datetime.time):
    extra = t.second % 3

    if extra != 0:
        return t.replace(second=t.second - extra)
    else:
        return t


def video_to_html(video):
    fig, ax = plt.subplots(1, 1)

    ax.set_axis_off()

    ims = [[ax.imshow(video[i], animated=True)] for i in range(len(video))]
    anim = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)

    plt.close()

    return anim.to_jshtml()

def videos_to_html(*videos) -> str:
    fig, ax = plt.subplots(1, len(videos))

    for i in range(len(videos)):
        ax[i].set_axis_off()

    ims = [[ax[j].imshow(videos[j][i], animated=True) for j in range(len(videos))] for i in range(len(videos[0]))]
    anim = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)

    plt.close()

    return anim.to_jshtml()

def videos_to_html_stack(*videos) -> str:
    fig, ax = plt.subplots(len(videos), 1)

    for i in range(len(videos)):
        ax[i].set_axis_off()

    ims = [[ax[j].imshow(videos[j][i], animated=True) for j in range(len(videos))] for i in range(len(videos[0]))]
    anim = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)

    plt.close()

    return anim.to_jshtml()

T = TypeVar('T')

class DisjointSet(Generic[T]):
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, value: T):
        if value not in self.parent:
            self.parent[value] = value
            self.rank[value] = 0
        elif self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])

    def find(self, value: T):
        if value not in self.parent:
            self.parent[value] = value
            self.rank[value] = 0

            return value

        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])

        return self.parent[value]

    def union(self, x: T, y: T):
        px, py = self.find(x), self.find(y)

        if px == py:
            return

        if self.rank[px] > self.rank[py]:
            self.parent[py] = px
        elif self.rank[px] < self.rank[py]:
            self.parent[px] = py
        else:
            self.parent[py] = px
            self.rank[px] += 1