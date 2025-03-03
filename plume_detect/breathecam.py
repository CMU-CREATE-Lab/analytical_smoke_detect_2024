import datetime
import math
from itertools import product
import numpy as np
import re
import requests

from typing import Callable, Union

from cmn import TIME_MACHINES, CAMERAS, decode_video_frames, get_camera_id, get_time
from motion import MotionAnalysis
from view import View

DateFormatter = Callable[[str], datetime.date]

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
                 nlevels: int = 1) -> np.ndarray:
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

        return cam.download_video(start_frame, frames, view or View.full(), nlevels)

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

    def analyze(self, 
                time: datetime.time, 
                view: Union[View, None] = None,
                frames: int = 1, *,
                nlevels: int = 1, 
                background_subtractor):

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

        view = view or View.full()
        video = self.download_video(time, frames, view, nlevels)
        
        return MotionAnalysis(video, background_subtractor=background_subtractor)
    
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

    def height(self, nlevels: int = 1) -> int:
        return int(math.ceil(self.r["height"] / nlevels))

    def level_from_subsample(self, nlevels: int) -> int:
        assert ((nlevels & (nlevels - 1)) == 0)

        level = self.levels - nlevels.bit_length()

        assert level >= 0, f"Subsample {nlevels} is too high for timemachine with {self.levels} levels."

        return level

    def subsample_from_level(self, level: int) -> int:
        assert (level > 0) and ((self.levels - level) > 0)

        return 2 ** (self.levels - level - 1)

    def tile_url(self, level: int, tile_x: int, tile_y: int) -> str:
        return f"{self.tile_root_url}/{level}/{tile_y * 4}/{tile_x * 4}.mp4"

    def width(self, nlevels: int = 1) -> int:
        return int(math.ceil(self.r["width"] / nlevels))
    