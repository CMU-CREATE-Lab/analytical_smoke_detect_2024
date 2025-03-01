from flask import Flask, render_template, request, jsonify
import re
import json
import psycopg2
from psycopg2.extras import DictCursor
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

connected_clients = {}

def get_db_connection():
    """Create a database connection to the PostgreSQL database"""
    conn = psycopg2.connect(dbname="smoke_detect")
    conn.autocommit = True
    return conn

def get_videos_for_run(run_name):
    """Get all videos for a specific run from the database"""
    videos = []
    classifications = {}
    
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=DictCursor)
    
    try:
        cur.execute(
            "SELECT * FROM video_labels WHERE run_name = %s ORDER BY id",
            (run_name,)
        )
        
        for index, row in enumerate(cur.fetchall()):
            # Parse metadata to extract video details
            metadata = row['metadata']
            if index == 0:
                classification_options = row['classifications']
            classifications = row['classifications']
            record_id = row['id']
            
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
                    "label_text": metadata,
                    "classifications": classifications,
                    "record_id": record_id
                }
                videos.append(video_data)
            
    except Exception as e:
        print(f"Database error: {e}")
        
    finally:
        cur.close()
        conn.close()
    
    # Sort videos by their number
    videos.sort(key=lambda v: v["number"])
    
    return (videos,classification_options)

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
    videos, classification_options = get_videos_for_run(run_name)
    
    if not videos:
        return render_template('error.html', 
                              message=f"No videos found for run: {run_name}"), 404
    
    return render_template('videos.html', 
                          videos=videos, 
                          run_name=run_name,
                          classification_options=classification_options)


@socketio.on('connect')
def handle_connect():
    print('Client connected:', request.sid)
    connected_clients[request.sid] = request

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected:', request.sid)
    if request.sid in connected_clients:
        del connected_clients[request.sid]

# @app.route('/update-video-classifications', methods=['POST'])
@socketio.on('update-video-classifications')
def set_classification_state(data):
    try:
        #raw_data = request.data
        #data = json.loads(raw_data)
        id = data.get('id')
        classifications_data = json.dumps(data.get('classifications'))

        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=DictCursor)

        cur.execute(
            "UPDATE video_labels SET classifications = %s WHERE id = %s",
            (classifications_data, id)
        )
        conn.commit()

        for sid in connected_clients:
            if sid != request.sid:
                socketio.emit('checkbox-update', {'id': id, 'classifications': json.loads(classifications_data)}, room=sid)

        return jsonify({'message': 'Video classifications updated successfully'}), 200
    except Exception as e:
        # Handle unexpected errors
        print(f"Error: {e}")
        return jsonify({'message': f'Error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
