"""
Database configuration for the plume_detect project.
"""

# Configuration for Unix socket connection (Ubuntu)
DB_CONFIG = {
    'dbname': 'smoke_detect',
    # No user, password, host needed for Unix socket connection
}

def get_db_connection_string():
    """Return a connection string for SQLAlchemy."""
    return f"postgresql:///smoke_detect"  # Local Unix socket connection

def get_db_connection():
    """Return a psycopg2 connection object."""
    import psycopg2
    return psycopg2.connect(**DB_CONFIG)
