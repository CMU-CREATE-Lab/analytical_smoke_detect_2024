from functools import cache
import os
from pathlib import Path
import dateutil
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import datetime
import subprocess
import numpy as np

import pytz

from stopwatch import Stopwatch
from thumbnail_api import BreathecamThumbnail
from timemachine import TimeMachine

class Thumbnails:
    @cache
    @staticmethod
    def clairton():
        # Natisha's Clairton view
        url = "https://share.createlab.org/shorturl/breathecam/f263bebc7632efa9"
        thumbnail = BreathecamThumbnail(url)

        # Increase resolution to 1:1
        thumbnail.set_scale(1, 1)

        # Increase size to 4K
        thumbnail.resize_rect_preserving_scale(3840, 2160)

        return thumbnail




@cache
def client():
    scope = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name("secrets/createlab-breathecam-bulk-video-generation-58427be4b55f.json", scope)
    client = gspread.authorize(creds)
    return client

required_columns = ["Site", "Date", "Begin time", "End time", "Video", "Notes"]
first_col = "A"
last_col = chr(ord("A") + len(required_columns) - 1)
# Use the "exports" subdirectory relative to this script's directory
export_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")

class BatchVideoExporter:
    def __init__(self, spreadsheet_name):
        self.df = self.read_spreadsheet(spreadsheet_name)

    def read_spreadsheet(self, spreadsheet_name):
        # Open the main sheet in this spreadsheet.  Complain if there's more than one sheet
        spreadsheet = client().open(spreadsheet_name)
        worksheets = spreadsheet.worksheets()
        if len(worksheets) > 1:
            raise ValueError(f"Expected only one sheet in {spreadsheet_name}, but found {len(worksheets)} sheets")
        worksheet = worksheets[0]

        # Get all tables in the worksheet    df = pd.DataFrame(data[1:], columns=data[0])
        rows = worksheet.get(f"{first_col}:{last_col}")

        header = rows[0]
        data_rows = rows[1:]
        assert header == required_columns

        df = pd.DataFrame(data_rows, columns=header)
        # None in Video or Notes columns should be empty string instead
        df["Video"] = df["Video"].fillna("")
        df["Notes"] = df["Notes"].fillna("")

        return df
    
    def export_video(self, row):
        site = row["Site"]
        # Parse as datetime.date
        date = dateutil.parser.parse(row["Date"]).date()
        # Parse as datetime.time
        begin_time = dateutil.parser.parse(row["Begin time"]).time()
        end_time = dateutil.parser.parse(row["End time"]).time()
        video = row["Video"]
        notes = row["Notes"]

        et = pytz.timezone("America/New_York")
        begin_datetime = et.localize(datetime.datetime.combine(date, begin_time))
        end_datetime = et.localize(datetime.datetime.combine(date, end_time))


        os.makedirs(export_directory, exist_ok=True)
        # Create temporary file in exports directory
        export_filename = site
        export_filename += f"-{begin_datetime.strftime('%Y%m%d-%H%M%S')}"
        export_filename += f"-{end_datetime.strftime('H%M%S')}-et"
        export_filename += ".mp4"
        export_path = os.path.join(export_directory, export_filename)

        with Stopwatch(f"Exporting video for {site} from {begin_datetime} to {end_datetime}"):
            render_video(site, begin_datetime, end_datetime, export_path)
            print(f"BatchVideoExporter: Exported video to {export_path} ({os.path.getsize(export_path)/1e6:.06f} MB)")
        
def render_video(site: str, begin_datetime: datetime.datetime, end_datetime: datetime.datetime, export_path: str):
    site = site.lower()
    # Assert begin and end have timezones
    assert begin_datetime.tzinfo is not None, "begin_datetime must have a timezone"
    assert end_datetime.tzinfo is not None, "end_datetime must have a timezone"
    # Assert site matches an attribute in Thumbnails
    assert hasattr(Thumbnails, site), f"Invalid site: {site}"
    thumbnail = getattr(Thumbnails, site)().copy()
    thumbnail.set_begin_end_times(begin_datetime, end_datetime)
    assert thumbnail.scale() == (1, 1), "Thumbnail must have a scale of 1:1"

    timemachine = TimeMachine.from_breathecam_thumbnail(thumbnail)

    # TO DO: Download multiple shards in parallel

    # Frames are ndarrays of shape (nframes, height, width, 3) with dtype uint8
    frames = timemachine.download_video_time_range(begin_datetime, end_datetime, thumbnail.view_rect(), subsample=1)
    print(f"BatchVideoExporter: Downloaded {len(frames)} frames")

    # Create temporary filename with pid and thread id
    temp_path = f"{export_path}.{os.getpid()}.{id(frames)}.mp4"
    
    # Remove temporary file if it exists
    Path(temp_path).unlink(missing_ok=True)

    # Encode frames into mp4 using external ffmpeg process
    height, width = frames[0].shape[:2]
    command = [
        'ffmpeg',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',  # size of one frame
        '-pix_fmt', 'rgb24',
        '-r', '30',  # frames per second
        '-i', '-',  # The input comes from a pipe
        '-c:v', 'libx264',
        '-preset', 'slow',  # Higher quality encoding
        '-crf', '23',  # Constant Rate Factor (0-51, lower is better quality)
        '-pix_fmt', 'yuv420p',  # Compatibility for media players
        temp_path
    ]
    
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    
    for frame in frames:
        process.stdin.write(frame.tobytes())
    
    process.stdin.close()
    process.wait()
    
    if process.returncode == 0:
        # Atomic rename of temp file to final path
        os.replace(temp_path, export_path)
    else:
        # Clean up temp file if ffmpeg failed
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise RuntimeError(f"ffmpeg failed with return code {process.returncode}")



