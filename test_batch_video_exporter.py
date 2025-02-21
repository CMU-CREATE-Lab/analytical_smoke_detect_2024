import pytest
from batch_video_exporter import BatchVideoExporter, render_video
from datetime import datetime
from stopwatch import Stopwatch
from zoneinfo import ZoneInfo

def test_export_first_video():
    exporter = BatchVideoExporter("Natisha BreatheCam video exports")
    exporter.export_video(exporter.df.iloc[0])

def test_export_second_video():
    exporter = BatchVideoExporter("Natisha BreatheCam video exports")
    exporter.export_video(exporter.df.iloc[1])

def test_export_third_video():
    # takes 143 seconds for 1h of video
    # 214 with 5 chunk threads
    exporter = BatchVideoExporter("Natisha BreatheCam video exports")
    exporter.export_video(exporter.df.iloc[2])

def test_export_fourth_video():
    exporter = BatchVideoExporter("Natisha BreatheCam video exports")
    exporter.export_video(exporter.df.iloc[3])

def test_export_next_video():
    exporter = BatchVideoExporter("Natisha BreatheCam video exports")
    exporter.export_next()

def test_noop():
    print("hello from test_noop")

def test_export_edgar_thomson_south():
    # Create start time as 2/1/2025 8am eastern time
    # Create end time as 2/1/2025 8:05am easter
    begin_datetime = datetime(2025, 2, 1, 8, 0, 0, tzinfo=ZoneInfo("America/New_York"))
    end_datetime = datetime(2025, 2, 1, 8, 5, 0, tzinfo=ZoneInfo("America/New_York"))

    render_video("Edgar Thomson South", 
                 begin_datetime, end_datetime, 
                 "test_export_edgar_thomson_south.mp4")

    
