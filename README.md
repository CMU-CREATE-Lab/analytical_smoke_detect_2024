

Create virtual environment version 3.11

On hal52:
    /usr/bin/python3.11 -m venv venv && source venv/bin/activate
    pip install --upgrade pip && pip install -r requirements.txt




# Video labeler

A Flask application to view videos from plume detection runs, retrieving data from PostgreSQL.

## Setup

1. Install dependencies:
   ```
   pip install flask psycopg2-binary
   ```

2. Place jQuery in the static/js directory or update the template to use a CDN.

3. Make sure you have a PostgreSQL database named "smoke_detect" with a table "video_labels".
   The table should have the following schema:
   ```sql
   CREATE TABLE video_labels (
       id SERIAL PRIMARY KEY,
       run_name TEXT NOT NULL,
       video_url TEXT NOT NULL,
       metadata TEXT,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```

4. Run the application:
   ```
   python app.py
   ```

5. Access the application at:
   ```
   http://localhost:5000/
   ```

## Structure

- `app.py`: Main Flask application with PostgreSQL database access
- `templates/index.html`: Homepage showing available runs
- `templates/videos.html`: Page displaying videos for a specific run
- `templates/error.html`: Error page for when runs are not found
- `static/js/`: JavaScript files (including jQuery)
