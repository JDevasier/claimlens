import sqlite3
import numpy as np

# Function to create the database and tables
def create_sample_database(db_name="../data/sample.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create the bills table
    cursor.execute('''CREATE TABLE IF NOT EXISTS bills (
                      BillID INTEGER PRIMARY KEY,
                      BillSummary TEXT)''')

    # Create the Rollcalls table
    cursor.execute('''CREATE TABLE IF NOT EXISTS Rollcalls (
                      RollcallID INTEGER PRIMARY KEY,
                      BillID INTEGER,
                      MemberID TEXT,
                      VoteType TEXT,
                      FOREIGN KEY (BillID) REFERENCES bills(BillID))''')

    # Create the Members table
    cursor.execute('''CREATE TABLE IF NOT EXISTS Members (
                      BioGuideID TEXT PRIMARY KEY,
                      Wiki TEXT,
                      Name TEXT)''')

    # Insert some sample data into bills
    bills_data = [
        (1, "An act to improve education standards."),
        (2, "A bill to reduce carbon emissions."),
        (3, "Legislation to provide healthcare for all."),
    ]
    cursor.executemany("INSERT INTO bills (BillID, BillSummary) VALUES (?, ?)", bills_data)

    # Insert some sample data into Rollcalls
    rollcall_data = [
        (1, 1, 'A00001', 'Yea'),
        (2, 2, 'A00001', 'Nay'),
        (3, 3, 'A00002', 'Yea'),
    ]
    cursor.executemany("INSERT INTO Rollcalls (RollcallID, BillID, MemberID, VoteType) VALUES (?, ?, ?, ?)", rollcall_data)

    # Insert some sample data into Members
    members_data = [
        ('A00001', "John Smith", "John Smith"),
        ('A00002', "Jane Doe", "Jane Doe"),
    ]
    cursor.executemany("INSERT INTO Members (BioGuideID, Name, Wiki) VALUES (?, ?, ?)", members_data)

    conn.commit()
    conn.close()
    print("Database created and sample data inserted.")

# Run database setup
if __name__ == "__main__":
    create_sample_database()
