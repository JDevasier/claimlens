import sqlite3
import logging

def generate_query(fes: dict, bill: int, agent: str) -> str:
    """
    Generate the SQL query based on frame elements (FEs).
    
    Parameters:
    - fes (dict): Frame elements containing Agent, Issue, and Position.
    - bill (int): Bill ID.
    - agent (str): Agent (congress member).
    
    Returns:
    - str: SQL query string.
    """
    # Ensure required inputs are present
    if bill is None or agent is None:
        raise ValueError("Both 'bill' and 'agent' must be provided")

    # Generate query based on available data
    if fes.get("Position") is None:
        query = """
            SELECT * 
            FROM rollcalls 
            JOIN votes ON rollcalls.id = votes.rollcall_id 
            JOIN members ON votes.member_id = members.id 
            WHERE members.bioname = ? AND rollcalls.bill_id = ?
        """
    else:
        query = """
            SELECT * 
            FROM Votes 
            WHERE memberId = ? AND billID = ? 
            ORDER BY congressNumber DESC
        """
    return query

def query_database(db: sqlite3.Connection, query: str, params: tuple = ()):
    """
    Execute a query against the database and return the results.
    
    Parameters:
    - query (str): SQL query string.
    - db (sqlite3.Connection): Active SQLite database connection.
    - params (tuple): Query parameters to prevent SQL injection.

    Returns:
    - list: Query result rows.
    """
    cur = db.cursor()  # No 'with' statement here
    cur.execute(query, params)
    result = cur.fetchall()
    return result


def connect_to_db(db_file: str) -> sqlite3.Connection:
    """
    Connect to an SQLite database.

    Parameters:
    - db_file (str): Path to the SQLite database file.

    Returns:
    - sqlite3.Connection: Active database connection.
    """
    try:
        logging.info(f"Connecting to database: {db_file}")
        return sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(f"Failed to connect to database: {e}")
        return None

def load_congressmembers(db: sqlite3.Connection) -> dict:
    """
    Load all congress members and their bioguide IDs from the database.

    Parameters:
    - db (sqlite3.Connection): Active SQLite database connection.

    Returns:
    - dict: Dictionary with congress member names as keys and bioguide IDs as values.
    """
    congressmembers = {}
    query = "SELECT bioguideid, name FROM members;"
    results = query_database(db, query)

    for bioguideid, name in results:
        congressmembers[name] = bioguideid

    return congressmembers