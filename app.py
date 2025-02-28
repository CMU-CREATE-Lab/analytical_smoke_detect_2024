from flask import Flask, render_template
import os
import re
import json
import psycopg2
from psycopg2.extras import DictCursor
from collections import defaultdict

app = Flask(__name__)

def get_db_connection():
    """Create a database connection to the PostgreSQL database"""
    conn = psycopg2.connect(dbname="smoke_detect")
    conn.autocommit = True
    return conn

def get_videos_for_run(run_name):
    """Get all videos for a specific run from the database"""
    videos = []
    
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=DictCursor)
    
    try:
        cur.execute(
            "SELECT * FROM video_labels WHERE run_name = %s ORDER BY id",
            (run_name,)
        )
        
        for row in cur.fetchall():
            # Parse metadata to extract video details
            metadata = row['metadata']
            
            # Extract video number
            video_num = re.search(r'Video (\d+)', metadata)
            # Extract frames info
            frames_info = re.search(r'(\d+) frames starting at (\d+)', metadata)
            # Extract size info
            size_info = re.search(r'Size: (\d+)x(\d+)', metadata)
            # Extract points info
            points_info = re.search(r'Points: (\d+)', metadata)
            # Extract area info
            area_info = re.search(r'Area: (\d+)', metadata)
            # Extract density info
            density_info = re.search(r'Density: ([0-9.]+)', metadata)
            # Extract white pixels info
            white_pixels = re.search(r'White Pixels: (\d+)', metadata)
            
            if video_num and frames_info and size_info and points_info and area_info and density_info:
                video_data = {
                    "src": row['video_url'],
                    "number": int(video_num.group(1)),
                    "frames": int(frames_info.group(1)),
                    "start_frame": int(frames_info.group(2)),
                    "width": int(size_info.group(1)),
                    "height": int(size_info.group(2)),
                    "points": int(points_info.group(1)),
                    "area": int(area_info.group(1)),
                    "density": float(density_info.group(1)),
                    "white_pixels": int(white_pixels.group(1)) if white_pixels else 0,
                    "label_text": metadata
                }
                videos.append(video_data)
            
    except Exception as e:
        print(f"Database error: {e}")
        
    finally:
        cur.close()
        conn.close()
    
    # Sort videos by their number
    videos.sort(key=lambda v: v["number"])
    
    return videos

def get_available_runs():
    """Get a list of all available run names in the database"""
    runs = []
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("SELECT DISTINCT run_name FROM video_labels ORDER BY run_name")
        runs = [row[0] for row in cur.fetchall()]
    except Exception as e:
        print(f"Database error: {e}")
    finally:
        cur.close()
        conn.close()
        
    return runs

@app.route('/')
def index():
    """Show a list of available runs"""
    runs = get_available_runs()
    return render_template('index.html', runs=runs)

@app.route('/label/<run_name>')
def show_videos(run_name):
    """Show videos for a specific run"""
    videos = get_videos_for_run(run_name)
    
    if not videos:
        return render_template('error.html', 
                              message=f"No videos found for run: {run_name}"), 404
    
    return render_template('videos.html', 
                          videos=videos, 
                          run_name=run_name)

if __name__ == '__main__':
    app.run(debug=True)
