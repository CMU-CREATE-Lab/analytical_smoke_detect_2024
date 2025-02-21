# coded in part with https://claude.ai/chat/2fba7438-0e73-41d5-bd98-313e5d0a57cc

import datetime
import concurrent
import requests
from thumbnail_api import Rectangle, BreathecamThumbnail
import math
import numpy as np
import pandas as pd
from video_decoder import decode_video_frames
import pytz
import dateutil.parser
from functools import cache
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor


CAMERAS = {
    "Clairton Coke Works": "clairton4", 
    "Shell Plastics West": "vanport3", 
    "Edgar Thomson South": "westmifflin2",
    "Metalico": "accan2", 
    "Revolution ETC/Harmon Creek Gas Processing Plants": "cryotm",
    "Riverside Concrete": "cementtm", 
    "Shell Plastics East": "center1", 
    "Irvin": "irvin1", 
    "North Shore": "heinz", 
    "Mon. Valley": "walnuttowers1", 
    "Downtown": "trimont1", 
    "Oakland": "oakland"
}

class TimeMachine:
    def __init__(self, root_url: str, timezone=pytz.timezone("America/New_York")):
        self.root_url = root_url
        self.tm_url = f"{root_url}/tm.json"
        print(f"Fetching {self.tm_url}")
        self.tm = requests.get(self.tm_url).json()
        datasets = self.tm['datasets']
        assert(len(datasets) == 1)
        dataset = datasets[0]
        id = dataset['id']
        self.tile_root_url = f"{root_url}/{id}"
        self.r_url = f"{self.tile_root_url}/r.json"
        print(f"Fetching {self.r_url}")
        self.r = requests.get(self.r_url).json()
        print(f'TimeMachine has {self.r["nlevels"]} levels and {len(self.capture_times())} frames')
        self.timezone = timezone

    @cache
    def capture_datetimes(self):
        # Use strptime to parse the capture times since it's faster.  Format is 2024-09-09 00:00:00
        return [self.timezone.localize(datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S")) for t in self.capture_times()]
        #return [self.timezone.localize(dateutil.parser.parse(t)) for t in self.capture_times()]
    
    @cache
    def frameno_from_date_before_or_equal(self, dt: datetime.datetime):
        for i, t in enumerate(self.capture_datetimes()):
            if t > dt:
                return max(0, i - 1)
        return len(self.capture_datetimes()) - 1
    
    @cache
    def frameno_from_date_after_or_equal(self, dt: datetime.datetime):
        for i, t in enumerate(self.capture_datetimes()):
            if t >= dt:
                return i
        return len(self.capture_datetimes()) - 1
    
    @classmethod
    def from_breathecam_thumbnail(cls, thumbnail: BreathecamThumbnail):
        return cls(thumbnail.timemachine_root_url())

    @staticmethod
    def download(
        location: str, 
        date: datetime.date,
        time: datetime.time,
        frames: int, 
        rect: Rectangle, 
        subsample: int) -> np.ndarray:
        """
        Downloads a video for the given tmera location, date and time.

        Parameters:
        ---
        * location - Location of the tmera, refer to `CAMERAS` for valid locations.
        * date - The day of the video
        * time - The time to start the capture
        * frames - The number of frames to capture
        * rect - The view to capture
        * subsample - The subsample of the produced video

        Returns:
        ---
        If frames is 1, a numpy array of dimensions width*height*4. If frames
        is greater than 1, a numpy array of dimensions frames*width*height*4.
        """

        if date is None:
            raise Exception("Date not set.")

        if time is None:
            raise Exception("Time not set.")

        date_str = date.strftime("%Y-%m-%d")
        time_str = get_time(time)
        start_time = f"{date_str} {time_str.strftime('%H:%M:%S')}"
        url = f"{BASE_URL}/{CAMERAS[location]}/{date_str}.timemachine"

        tm = TimeMachine(url)

        start_frame = tm.frame_from_date(start_time)
        
        if start_frame < 0:
            raise Exception("First frame invalid.")
            return None
        
        remaining_frames = len(tm.capture_times()) - start_frame

        if remaining_frames < frames:
            frames = remaining_frames

        video = tm.download_video(start_frame, frames, view, subsample)

        opacity = np.full((video.shape[0], video.shape[1], video.shape[2], 1), 255, dtype=video.dtype)
        video = np.concatenate((video, opacity), axis=3) / 255.0

        if frames == 1:
            return video[0]
        else:
            return video

    def download_video_frame_range(self, start_frame_no: int, nframes: int, rect: Rectangle, subsample:int=1, max_threads:int=8):
        """
        Download and assemble video tiles into a single numpy array using parallel threads.
        
        Args:
            start_frame_no: Starting frame number
            nframes: Number of frames to download
            rect: Rectangle coordinates after subsampling
            subsample: Subsample factor
            max_threads: Maximum number of concurrent download threads
        
        Returns:
            numpy.ndarray: Array of shape (nframes, height, width, 3) containing the video data
        """
        
        rect = rect.ensure_integer()
        level = self.level_from_subsample(subsample)
        level_width = self.width(subsample)
        level_height = self.height(subsample)
        
        # Create output array
        result = np.zeros((nframes, rect.height, rect.width, 3), dtype=np.uint8)
        
        # Compute tile range
        min_tile_y = rect.y1 // self.tile_height()
        max_tile_y = 1 + (rect.y2 - 1) // self.tile_height()
        min_tile_x = rect.x1 // self.tile_width()
        max_tile_x = 1 + (rect.x2 - 1) // self.tile_width()
        
        # Function to download and process a single tile
        def process_tile(tile_x, tile_y):
            tile_url = self.tile_url(level, tile_x, tile_y)
            
            # # Check if tile exists
            # response = requests.head(tile_url)
            # if response.status_code == 404:
            #     return None
                
            tile_rectangle = Rectangle(
                tile_x * self.tile_width(),
                tile_y * self.tile_height(),
                (tile_x + 1) * self.tile_width(),
                (tile_y + 1) * self.tile_height()
            )
            
            intersection = rect.intersection(tile_rectangle)
            if intersection is None:
                return None
                
            src_rect = intersection.translate(-tile_rectangle.x1, -tile_rectangle.y1)
            dest_rect = intersection.translate(-rect.x1, -rect.y1)
            
            try:
                frames = decode_video_frames(
                    video_url=tile_url,
                    start_frame=start_frame_no,
                    n_frames=nframes,
                    width = self.tile_width(),
                    height = self.tile_height(),
                    fps = self.fps()
                )
                return (frames, src_rect, dest_rect)
            except Exception as e:
                print(f"Error processing tile {tile_url}: {str(e)}")
                return None

        # Create list of all tile coordinates
        tiles = [(x, y) for y in range(min_tile_y, max_tile_y) 
                       for x in range(min_tile_x, max_tile_x)]
        n_tiles = len(tiles)
        print(f"Processing {n_tiles} tiles with {max_threads} threads")

        # Process tiles in parallel
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            # Submit all tasks
            future_to_tile = {executor.submit(process_tile, x, y): (x, y) 
                            for x, y in tiles}
            
            # Process results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(future_to_tile), 1):
                tile_x, tile_y = future_to_tile[future]
                result_data = future.result()
                
                if result_data is not None:
                    frames, src_rect, dest_rect = result_data
                    result[:, 
                           dest_rect.y1:dest_rect.y2,
                           dest_rect.x1:dest_rect.x2,
                           :] = frames[:, 
                                     src_rect.y1:src_rect.y2,
                                     src_rect.x1:src_rect.x2,
                                     :]
                print(f"Completed {i} of {n_tiles} tiles")

        return result

    def download_video_time_range(self, start_time: datetime.datetime, end_time: datetime.datetime, rect: Rectangle, subsample:int=1):   
        start_frame = self.frameno_from_date_after_or_equal(start_time)
        end_frame = self.frameno_from_date_before_or_equal(end_time)
        nframes = end_frame - start_frame + 1
        return self.download_video_frame_range(start_frame, nframes, rect, subsample)

    # tile_x and tile_y are in tile coordinates / 4
    def tile_url(self, level:int, tile_x:int, tile_y:int):
        return f"{self.tile_root_url}/{level}/{tile_y*4}/{tile_x*4}.mp4"

    def level_from_subsample(self, subsample:int) -> int:
        log2_subsample = math.log2(subsample)
        assert(log2_subsample.is_integer())
        # Find level_info for subsample
        level_number = round(len(self.level_info()) - 1 - log2_subsample)
        print(f"Subsample {subsample} corresponds to level {level_number}")
        assert level_number >= 0, f"Subsample {subsample} too high for timemachine of {len(self.level_info())} levels (max subsample {self.max_subsample()})"
        return level_number
    
    def subsample_from_level(self, level:int) -> int:
        return 2 ** (len(self.level_info()) - 1 - level)

    def max_subsample(self) -> int:
        return self.subsample_from_level(0)

    # Convenience accessors for tm and r
    def capture_times(self):
        return self.tm["capture-times"]
    
    def level_info(self):
        return self.r["level_info"]
    
    def fps(self):
        return self.r["fps"]
    
    def width(self, subsample:int=1):
        return int(math.ceil(self.r["width"]/subsample))
    
    def height(self, subsample:int=1):
        return int(math.ceil(self.r["height"]/subsample))

    def tile_width(self):
        return self.r["video_width"]
    
    def tile_height(self):
        return self.r["video_height"]
    
    def info(self):
        print(f"TimeMachine root: {self.root_url}")
        print(f"Tile root: {self.tile_root_url}")
        print(f"Capture times: {self.capture_times()}")
        print(f"Level info: {self.level_info()}")
        print(f"FPS: {self.fps()}")
        print(f"Width: {self.width()}")
        print(f"Height: {self.height()}")
        print(f"Tile width: {self.tile_width()}")
        print(f"Tile height: {self.tile_height()}")
        print(f"r: {self.r}")
        print(f"tm: {self.tm}")   