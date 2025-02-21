from functools import cache
import os
from pathlib import Path
import threading
import dateutil
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import datetime
import subprocess
import numpy as np
import argparse

import pytz

from stopwatch import Stopwatch
from thumbnail_api import BreathecamThumbnail
from timemachine import TimeMachine
from concurrent.futures import ThreadPoolExecutor
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

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
        self.spreadsheet_name = spreadsheet_name
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
        export_filename += f"-{end_datetime.strftime('%H%M%S')}-et"
        export_filename += ".mp4"
        export_path = os.path.join(export_directory, export_filename)

        with Stopwatch(f"Exporting video for {site} from {begin_datetime} to {end_datetime}"):
            render_video(site, begin_datetime, end_datetime, export_path)
            print(f"BatchVideoExporter: Exported video to {export_path} ({os.path.getsize(export_path)/1e6:.06f} MB)")

        return export_path

    def find_next_row(self):
        """Find the first row where Video column is empty and all required fields are present"""
        empty_video_mask = self.df["Video"] == ""
        valid_fields_mask = (
            self.df["Site"].notna() &
            self.df["Date"].notna() & 
            self.df["Begin time"].notna() &
            self.df["End time"].notna()
        )
        eligible_rows = self.df[empty_video_mask & valid_fields_mask]
        if len(eligible_rows) == 0:
            return None
        return eligible_rows.iloc[0]

    def update_spreadsheet_cell(self, row_idx, value):
        """Update the Video cell for the given row, but verify row contents first"""
        worksheet = client().open(self.spreadsheet_name).worksheets()[0]
        # Spreadsheet rows are 1-based and include header
        sheet_row = row_idx + 2
        
        # Read the entire row to verify contents
        row_data = worksheet.row_values(sheet_row)
        expected_row = self.df.iloc[row_idx]
        
        # Verify key fields match
        if (row_data[0] != expected_row["Site"] or
            row_data[1] != expected_row["Date"] or
            row_data[2] != expected_row["Begin time"] or
            row_data[3] != expected_row["End time"]):
            raise ValueError(
                f"Row contents changed while processing! Expected:\n"
                f"Site: {expected_row['Site']}, Date: {expected_row['Date']}, "
                f"Begin: {expected_row['Begin time']}, End: {expected_row['End time']}\n"
                f"But found:\n"
                f"Site: {row_data[0]}, Date: {row_data[1]}, "
                f"Begin: {row_data[2]}, End: {row_data[3]}"
            )
        
        # If verification passes, update the cell
        # Update using row/col numbers instead of A1 notation
        worksheet.update_cell(sheet_row, 5, value)  # 5 is the column number for "Video" (E)
        # Update local dataframe
        self.df.at[row_idx, "Video"] = value

    def upload_to_drive(self, file_path):
        """Upload file to Google Drive and make it world-readable"""
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            "secrets/createlab-breathecam-bulk-video-generation-58427be4b55f.json",
            ['https://www.googleapis.com/auth/drive']
        )
        drive_service = build('drive', 'v3', credentials=creds)
        
        file_metadata = {
            'name': os.path.basename(file_path),
            'parents': ['root']
        }
        media = MediaFileUpload(file_path, resumable=True)
        
        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id,webViewLink'
        ).execute()
        
        # Make the file world-readable
        drive_service.permissions().create(
            fileId=file['id'],
            body={'type': 'anyone', 'role': 'reader'},
            fields='id'
        ).execute()
        
        return file['webViewLink']

    def export_next(self):
        """Export the next video in the queue"""
        row = self.find_next_row()
        if row is None:
            print("No more videos to export")
            return False

        row_idx = row.name
        start_time = datetime.datetime.now(pytz.UTC).astimezone()
        self.update_spreadsheet_cell(row_idx, f"Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            export_path = self.export_video(row)

            # Upload to Google Drive
            video_url = self.upload_to_drive(export_path)
            video_link = f'=hyperlink("{video_url}", "{os.path.basename(export_path)}")'
            self.update_spreadsheet_cell(row_idx, video_link)
            return True

        except Exception as e:
            self.update_spreadsheet_cell(row_idx, f"Error: {str(e)}")
            raise
        
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

    # Create temporary filename with pid and thread id
    temp_path = f"{export_path}.{os.getpid()}.{threading.get_ident()}.mp4"
    
    # Remove temporary file if it exists
    Path(temp_path).unlink(missing_ok=True)

    # Encode frames into mp4 using external ffmpeg process
    width = int(thumbnail.view_rect().width)
    height = int(thumbnail.view_rect().height)
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
        '-crf', '18',  # Constant Rate Factor (0-51, lower is better quality)
        '-pix_fmt', 'yuv420p',  # Compatibility for media players
        temp_path
    ]
    
    process = subprocess.Popen(command, stdin=subprocess.PIPE)

    start_frame = timemachine.frameno_from_date_after_or_equal(begin_datetime)
    end_frame = timemachine.frameno_from_date_before_or_equal(end_datetime)
    nframes = end_frame - start_frame + 1

    # We have to process in small chunks or we run out of RAM
    chunk_size = 100
    frame_chunks = range(start_frame, start_frame + nframes, chunk_size)
    
    def download_chunk(chunk_info):
        chunk_start, chunk_frames = chunk_info
        frames = timemachine.download_video_frame_range(chunk_start, chunk_frames, thumbnail.view_rect(), subsample=1)
        print(f"BatchVideoExporter: Downloaded {len(frames)} frames")
        return frames

    # Create list of chunk information tuples
    chunk_infos = []
    for chunk_start in frame_chunks:
        chunk_frames = min(chunk_size, start_frame + nframes - chunk_start)
        chunk_infos.append((chunk_start, chunk_frames))

    # Multiple workers didn't seem to help here, so we use a single worker
    with ThreadPoolExecutor(max_workers=1) as executor:
        # Submit all jobs and store futures in order
        futures = list(executor.map(download_chunk, chunk_infos))

        # Write frames to ffmpeg process in correct order
        for frames in futures:
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

def main():
    parser = argparse.ArgumentParser(description='Batch Video Exporter for Breathecam')
    parser.add_argument('spreadsheet_name', help='Name of the Google Spreadsheet to process')
    parser.add_argument('--export-next', action='store_true',
                       help='Export the next pending video from the spreadsheet')
    
    args = parser.parse_args()
    
    exporter = BatchVideoExporter(args.spreadsheet_name)
    
    if args.export_next:
        try:
            if exporter.export_next():
                print("Successfully exported next video")
                return 0
            else:
                print("No videos pending export")
                return 1
        except Exception as e:
            print(f"Error exporting video: {str(e)}")
            return 2
    else:
        parser.print_help()
        return 1

if __name__ == '__main__':
    exit(main())



