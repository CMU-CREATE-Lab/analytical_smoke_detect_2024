#!/bin/bash

source venv/bin/activate

while true; do
    python batch_video_exporter.py "Natisha BreatheCam video exports" --export-next
    sleep 60
done

