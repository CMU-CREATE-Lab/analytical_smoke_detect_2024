import pytest
from batch_video_exporter import BatchVideoExporter
from stopwatch import Stopwatch

def test_export_first_video():
    exporter = BatchVideoExporter("Natisha BreatheCam video exports")
    exporter.export_video(exporter.df.iloc[0])

def test_export_second_video():
    exporter = BatchVideoExporter("Natisha BreatheCam video exports")
    exporter.export_video(exporter.df.iloc[1])

