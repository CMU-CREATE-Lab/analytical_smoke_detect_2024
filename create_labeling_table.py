#%%

from bs4 import BeautifulSoup
import os
import psycopg2

html_path = "/home/hjhawkinsiv/lib-plume-detect/events_1800_frames_MOG2_3.html"
html_base_dir = os.path.dirname(html_path)
html_page_name = os.path.splitext(os.path.basename(html_path))[0]
html = open(html_path, "r").read()

# Parse the HTML
soup = BeautifulSoup(html, 'html.parser')

# Find all video containers
video_containers = soup.find_all('div', class_='video-container')

base_url_path = "https://videos.breathecam.org/labeling/"
base_dir = "/workspace/projects/videos.breathecam.org/www/vids/labeling/"

results = []
for container in video_containers:  # Process all videos
    # Get video src and make it absolute
    video_src = container.find('video')['src']
    video_src = os.path.join(html_base_dir, video_src)
    # Follow symlinks
    video_src = os.path.realpath(video_src)
    # Assert that the path is within the base directory
    assert video_src.startswith(base_dir)
    # Change to URL by removing base directory and adding base url
    video_src = video_src[len(base_dir):]
    video_src = base_url_path + video_src
  
    # Get label text
    label_div = container.find('div', class_='label')
    # Replace <br> with spaces and get text
    for br in label_div.find_all('br'):
        br.replace_with(' ')
    metadata = label_div.text.strip()
    
    results.append({
        'run_name': html_page_name,
        'video_src': video_src,
        'metadata': metadata
    })

# Connect to PostgreSQL database using Unix socket
conn = psycopg2.connect(dbname="smoke_detect")
cursor = conn.cursor()

# Create table if it doesn't exist (with run_name field)

# Delete video_labels table
drop = False
if drop:
    cursor.execute("DROP TABLE IF EXISTS video_labels")


cursor.execute("""
CREATE TABLE IF NOT EXISTS video_labels (
    id SERIAL PRIMARY KEY,
    run_name TEXT NOT NULL,
    video_url TEXT NOT NULL,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

# Delete existing records with the same run_name
cursor.execute("DELETE FROM video_labels WHERE run_name = %s", (html_page_name,))
print(f"Deleted existing records for run_name: {html_page_name}")

# Insert data into the table
for item in results:
    cursor.execute(
        "INSERT INTO video_labels (run_name, video_url, metadata) VALUES (%s, %s, %s)",
        (item['run_name'], item['video_src'], item['metadata'])
    )

# Commit the transaction
conn.commit()
print(f"Successfully inserted {len(results)} video records into the database.")

# Close the connection
cursor.close()
conn.close()

# %%
