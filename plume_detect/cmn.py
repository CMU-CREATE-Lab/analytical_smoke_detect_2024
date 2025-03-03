import datetime
import ffmpeg
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


def make_video_div(src: str, metadata: list[str]) -> str:
   return f"""
<div class="video-container">
  <video src="{src}" loop muted controls playbackRate=0.5></video>
  <div class="label">{'<br>'.join(metadata)}</div>
</div>
"""


def write_mp4_video(path: str, video: np.ndarray):
    h, w = video.shape[1], video.shape[2]

    process = (
        ffmpeg
        .input("pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{w}x{h}")
        .filter("pad", width="ceil(iw/2)*2", height="ceil(ih/2)*2", color="black")
        .output(path, pix_fmt="yuv420p", vcodec="libx264", r=12, loglevel="quiet")
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for frame in video:
        process.stdin.write(frame.tobytes())

    process.stdin.close()
    process.wait()

def write_video_html(path: str, content: str, title: str = "") -> str:
  html = f"""
<html>
  <head>
    <script src="js/jquery-3.7.1.min.js"></script>
    <style>
      body {{margin:0px}}
      #site-title {{
        margin-bottom: 5px;
        text-align: center;
        width: 100%;
      }}
      .label {{
        font-size: 17px;
      }}
      .video-container {{
        display: inline-flex;
        flex-flow: column nowrap;
        margin: 3px;
      }}
    </style>
  </head>
  <body>
  {f'<h1 id="site-title">{title}</h1>' if (title := title.strip()) else ''}
  {content}

    <script>
      var ggg;
      function fixupVideo(v) {{
        console.log(v);
        var d = v.wrap('<div/>');
        console.log(d);
        if (v.attr('playbackRate')) {{
          v[0].playbackRate = parseFloat($(v).attr('playbackRate'));
        }}
        if (v.attr('trimRight')) {{
          vvv = v;
          ddd = d;
        }}
      }}

      function init() {{
        console.log('init');
        let observer = new IntersectionObserver(
          (entries, observer) => {{
            for (entry of entries) {{
              if (entry.isIntersecting) {{
            console.log('play', entry.target);
                entry.target.play();
              }} else  {{
          console.log('pause', entry.target);
                entry.target.pause(); 
        }}
            }}
          }},
          {{threshold: 0}}
        );
        
        $('img,video').each(function(i,v){{
          fixupVideo($(v));
          console.log('setting up', v);
          observer.observe(v);
        }});
      }}

      $(init);
    </script>
  </body>
</html>
"""
  with open(path, "w") as htmlFile:
    htmlFile.write(html)

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