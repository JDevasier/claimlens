import logging
import sqlite3
import argparse
from sentence_transformers import SentenceTransformer

def query_database(query, db, params=()):
    """Execute a query on the given database connection."""
    cur = db.cursor()  # No 'with' statement here
    cur.execute(query, params)
    result = cur.fetchall()
    return result


def connect_to_db(db_file: str) -> sqlite3.Connection:
    try:
        logging.info(f"Connecting to database: {db_file}")
        return sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(f"Failed to connect to database: {e}")
        return None

def get_all_bills(db_conn):
    """Retrieve all bill summaries from the database."""
    query = "SELECT BillID, BillSummary FROM bills;"
    bills = query_database(query, db_conn)
    if not bills:
        logging.error("No bills found in the database.")
    return bills

def embed_bills(bills, model):
    """Embed the bill summaries using the specified SentenceTransformer model."""
    bill_ids, summaries = zip(*bills)  # Separate bill IDs and summaries
    embeddings = model.encode(summaries, show_progress_bar=True)
    return dict(zip(bill_ids, embeddings))

def store_embeddings(db_conn, embeddings):
    """Store the generated embeddings in the database."""
    cursor = db_conn.cursor()

    # Create a new table to store embeddings if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS BillEmbeddings (
                      BillID INTEGER PRIMARY KEY,
                      Embedding BLOB)''')
    
    # Insert embeddings into the table
    for bill_id, embedding in embeddings.items():
        cursor.execute("INSERT OR REPLACE INTO BillEmbeddings (BillID, Embedding) VALUES (?, ?)", 
                       (bill_id, sqlite3.Binary(embedding.tobytes())))

    db_conn.commit()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Retrieve and embed bill summaries using a SentenceTransformer model.")
    parser.add_argument('--db', type=str, default='../data/sample.db', help='Path to the SQLite database (default: sample.db).')
    parser.add_argument('--model_name', type=str, default='msmarco-distilbert-base-tas-b', help='SentenceTransformer model to use (default: msmarco-distilbert-base-tas-b).')

    args = parser.parse_args()

    # Initialize logging
    logging.basicConfig(level=logging.INFO)

    # Connect to the database
    try:
        db_conn = connect_to_db(args.db)
        logging.info(f"Connected to database '{args.db}'")
    except Exception as e:
        logging.error(f"Failed to connect to database: {e}")
        return

    # Load the sentence transformer model
    try:
        model = SentenceTransformer(args.model_name)
        logging.info(f"Loaded model '{args.model_name}'")
    except Exception as e:
        logging.error(f"Failed to load the model '{args.model_name}': {e}")
        return

    # Retrieve all bills from the database
    bills = get_all_bills(db_conn)
    if not bills:
        return

    # Embed the bill summaries
    embeddings = embed_bills(bills, model)
    logging.info(f"Generated embeddings for {len(embeddings)} bills")

    # Store the embeddings in the database
    store_embeddings(db_conn, embeddings)
    logging.info("Stored embeddings in the database")

if __name__ == "__main__":
    main()
