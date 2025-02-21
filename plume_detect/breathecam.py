import datetime
import math
from functools import lru_cache
from itertools import product
import numpy as np
import re
import requests

from dataclasses import dataclass
from typing import Callable, Literal, Union

from .cmn import TIME_MACHINES, CAMERAS, decode_video_frames, get_camera_id, get_time
from .motion import Motion
from .view import View

DateFormatter = Callable[[str], datetime.date]

@dataclass
class BreatheCamMetadata:
    root_url: str
    tile_root: str
    day: str
    capture_times: list[str]
    levels: int
    level_info: list
    fps: int
    width: int
    height: int
    tile_width: int
    tile_height: int
    r: dict
    tm: dict


class BreatheCam:
    def __init__(self, root_url: str):
        day = re.search(r"\d\d\d\d-\d\d-\d\d", root_url)

        if day is None:
            raise Exception(f"Invalid root url: `{root_url}`.")

        self.day = day[0]
        self.root_url = root_url
        self.tm_url = f"{root_url}/tm.json"
        self.tm = requests.get(self.tm_url).json()

        datasets = self.tm["datasets"]

        assert len(datasets) == 1

        dataset = datasets[0]
        did = dataset["id"]

        self.tile_root_url = f"{root_url}/{did}"
        self.r_url = f"{self.tile_root_url}/r.json"
        self.r = requests.get(self.r_url).json()
        self.levels = self.r["nlevels"]

    @staticmethod
    def init_from(loc: str, day: Union[datetime.date, str], *, formatter: Union[DateFormatter, None] = None):
        if (loc_id := get_camera_id(loc)) is None:
            raise Exception(f"Invalid camera: {loc}.")

        if isinstance(day, str):
            day = (formatter or datetime.date.fromisoformat)(day)

        return BreatheCam(f"{TIME_MACHINES}/{loc_id}/{day.strftime('%Y-%m-%d')}.timemachine")

    @staticmethod
    def download(loc: str,
                 day: datetime.date,
                 time: datetime.time,
                 view: Union[View, None] = None,
                 frames: int = 1,
                 subsample: int = 1) -> np.ndarray:
        day_str = day.strftime("%Y-%m-%d")
        start_time = f"{day_str} {get_time(time).strftime('%H:%M:%S')}"
        url = f"{TIME_MACHINES}/{CAMERAS[get_camera_id(loc)]}/{day_str}.timemachine"
        cam = BreatheCam(url)
        start_frame = cam.capture_time_to_frame(start_time)

        if start_frame < 0:
            raise Exception("First frame invalid.")

        remaining_frames = len(cam.capture_times) - start_frame

        if remaining_frames < frames:
            frames = remaining_frames

        return cam.download_video(start_frame, frames, view or View.full(), subsample)

    @property
    def capture_times(self):
        return self.tm["capture-times"]

    @property
    def fps(self) -> int:
        return self.r["fps"]

    @property
    def level_info(self):
        return self.r["level_info"]

    @property
    def tile_height(self) -> int:
        return self.r["video_height"]

    @property
    def tile_width(self) -> int:
        return self.r["video_width"]

    # Coordinates:  The View (rectangle) is in full-resolution coords
    # Internal to this function, the view is modified to match the subsample as the internal
    # coords are divided by subsample
    def download_video(self,
                       start_frame_no: Union[int, datetime.time],
                       nframes: int,
                       view: Union[View, None] = None,
                       nlevels: int = 1) -> np.ndarray:
        
        if isinstance(start_frame_no, datetime.time):
            start_time = f"{self.day} {get_time(start_frame_no).strftime('%H:%M:%S')}"
            start_frame_no = self.capture_time_to_frame(start_time)

        if start_frame_no < 0 or start_frame_no >= len(self.capture_times):
            raise Exception("First frame invalid.")

        nframes = min(nframes, len(self.capture_times) - start_frame_no)
        view = (view or view.full()).subsample(nlevels)
        level = self.level_from_subsample(nlevels)
        result = np.zeros((nframes, view.height, view.width, 3), dtype=np.uint8)
        th, tw = self.tile_height, self.tile_width
        min_tile_y = view.top // th
        max_tile_y = 1 + (view.bottom - 1) // th
        min_tile_x = view.left // tw
        max_tile_x = 1 + (view.right - 1) // tw

        for tile_y, tile_x in product(range(min_tile_y, max_tile_y), range(min_tile_x, max_tile_x)):
            tile_url = self.tile_url(level, tile_x, tile_y)
            response = requests.head(tile_url)

            if response.status_code == 404:
                print(f"Warning: tile {tile_x},{tile_y} does not exist, skipping...")
                continue

            tile_view = View(tile_x * tw, tile_y * th, (tile_x + 1) * tw, (tile_y + 1) * th)

            intersection = view.intersection(tile_view)

            assert intersection, f"Tile ({tile_x}, {tile_y}) does not intersect view {view}"

            src_view = intersection.translate(-tile_view.left, -tile_view.top)
            dest_view = intersection.translate(-view.left, -view.top)

            try:
                # Download the tile video
                frames, metadata = decode_video_frames(tile_url, start_frame_no, nframes)

                # Copy the intersection region to the result array
                result[:, dest_view.top:dest_view.bottom, dest_view.left:dest_view.right, :] = (
                    frames[:, src_view.top:src_view.bottom, src_view.left:src_view.right, :])

            except Exception as e:
                print(f"Error processing tile {tile_url}: {str(e)}")
                continue

        return result

    def capture_time_to_frame(self, date: str) -> int:
        return self.tm["capture-times"].index(date)

    def height(self, subsample: int = 1) -> int:
        return int(math.ceil(self.r["height"] / subsample))

    def level_from_subsample(self, subsample: int) -> int:
        assert ((subsample & (subsample - 1)) == 0)

        level = self.levels - subsample.bit_length()

        assert level >= 0, f"Subsample {subsample} is too high for timemachine with {self.levels} levels."

        return level

    def metadata(self) -> BreatheCamMetadata:
        return BreatheCamMetadata(
            root_url=self.root_url,
            tile_root=self.tile_root_url,
            day=self.day,
            capture_times=self.capture_times(),
            levels=self.levels,
            level_info=self.level_info(),
            fps=self.fps,
            width=self.width(),
            height=self.height(),
            tile_width=self.tile_width,
            tile_height=self.tile_height,
            r=self.r,
            tm=self.tm
        )

    def subsample_from_level(self, level: int) -> int:
        assert (level > 0) and ((self.levels - level) > 0)

        return 2 ** (self.levels - level - 1)

    def tile_url(self, level: int, tile_x: int, tile_y: int) -> str:
        return f"{self.tile_root_url}/{level}/{tile_y * 4}/{tile_x * 4}.mp4"

    def width(self, subsample: int = 1) -> int:
        return int(math.ceil(self.r["width"] / subsample))


class MotionCapture:
    def __init__(self, 
                 loc: str, 
                 day: datetime.date, 
                 time: datetime.time, 
                 view: Union[View, None] = None,
                 frames: int = 1, 
                 subsample: int = 1, *,
                 detectShadows: bool = True,
                 varThreshold: int = 16):
        self.location = loc
        self.day = day
        self.time = time
        self.view = view or View.full()
        self.width = self.view.width
        self.height = self.view.height
        self.frames = frames
        self.subsample = subsample
        self.breathecam = BreatheCam.init_from(loc, day)
        self.motion = Motion(
            self.breathecam.download_video(time, frames + 1, view, subsample),
            detectShadows=detectShadows,
            varThreshold=varThreshold)

    def capture(self, subsample: int):
        if subsample == self.subsample:
            return self
        
        return MotionCapture(self.location, self.day, self.time, self.view, self.frames, subsample)
    
    def download_video(self, nlevels: int = 1):
        return self.breathecam.download_video(self.time, self.frames + 1, self.view, nlevels)

    def event_space(self, *, neighborhood: Literal[4, 8] = 8, span: int = 3):
        return self.motion.event_space(neighborhood=neighborhood, span=span)


class MotionDetector:
    def __init__(self, loc: str, day: datetime.date):
        self.location = loc
        self.day = day

    @lru_cache
    def analyze(self,
                time: datetime.time,
                view: Union[View, None] = None,
                frames: int = 1,
                subsample: int = 1, *,
                detectShadows: bool = True,
                varThreshold: int = 16):
        if time.second < 3:
            second = 57 + time.second

            if time.minute == 0:
                if time.hour == 0:
                    raise Exception(f"Invalid start frame: {self.day.strftime('%Y-%m-%d')} {time.strftime('%H:%M:%S')}")
                else:
                    t = time.replace(hour=time.hour - 1, minute=59, second=second)
            else:
                t = time.replace(minute = time.minute - 1, second = second)
        else:
            t = time.replace(second=time.second - 3)

        return MotionCapture(
            self.location, 
            self.day, 
            t, 
            view, 
            frames, 
            subsample, 
            detectShadows=detectShadows, 
            varThreshold=varThreshold)