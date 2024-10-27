import requests
from thumbnail_api import Rectangle
import math
import numpy as np
from video_decoder import decode_video_frames

class TimeMachine:
    def __init__(self, root_url: str):
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

    def download_video(self, start_frame_no: int, nframes: int, rect: Rectangle, subsample:int=1):
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
                assert intersection is not None, f"Tile {tile_x},{tile_y} does not intersect rectangle {rect}"
                
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