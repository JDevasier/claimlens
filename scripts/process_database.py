# Create database and insert into Congress, CongressMember, and Member tables

import os
import sqlite3
import csv
import sys
import re
import json
import xml.etree.ElementTree as ET

# Database file path
db_file = 'bills_congress.db'

# Check if the database file exists
if os.path.exists(db_file):
    # Remove the existing database file
    os.remove(db_file)
    print('Existing database dropped.')

# Create a connection to the database
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Create the Congress table
cursor.execute('''CREATE TABLE Congress (
                    CongressNumber INTEGER PRIMARY KEY NOT NULL
                )''')

# Create the Bills table
cursor.execute('''CREATE TABLE Bills (
                    BillID INTEGER PRIMARY KEY AUTOINCREMENT,
                    BillCongress INTEGER,
                    BillType TEXT,
                    BillNumber INTEGER,
                    BillDescription TEXT,
                    BillSummary TEXT,
                    FOREIGN KEY (BillCongress) REFERENCES Congress(CongressNumber)
                )''')

# Create the Members table
cursor.execute('''CREATE TABLE Members (
                    BioguideID TEXT PRIMARY KEY NOT NULL,
                    Name TEXT
                )''')

# Create the CongressMembers table ------------------------ MAKE PRIMARY NOT NULL AFTER YOU GET CONGRESSNUMBER
cursor.execute('''CREATE TABLE CongressMembers (
                    CongressNumber INTEGER,
                    BioguideID TEXT,
                    PRIMARY KEY (CongressNumber, BioguideID),
                    FOREIGN KEY (CongressNumber) REFERENCES Congress(CongressNumber),
                    FOREIGN KEY (BioguideID) REFERENCES Members(BioguideID)
                )''')

# Create the Rollcalls table
cursor.execute('''CREATE TABLE Rollcalls (
                    RollcallID TEXT PRIMARY KEY NOT NULL,
                    MemberID TEXT,
                    VoteType TEXT,
                    BillID INTEGER,
                    FOREIGN KEY (MemberID) REFERENCES Members(BioguideID),
                    FOREIGN KEY (BillID) REFERENCES Bills(BillID)
                )''')

csv.field_size_limit(sys.maxsize)

# Convert year to congress number
def convert_year_to_congress(year):
    return int((year - 1789) / 2 + 1)

# Insert data from legislators-current.json into Members and CongressMembers tables
with open('../data/vote/legislators-current.json') as json_file:
    data = json.load(json_file)
    line_count = 0
    for row in data:
        bioguide_id = row["id"]["bioguide"]
        name = row["name"]["official_full"]
        congress_number = convert_year_to_congress(int(row["terms"][0]["start"][:4]))  # Assuming the first term is used for congress number

        # Insert data into the Members table
        cursor.execute('''INSERT INTO Members (BioguideID, Name)
                        VALUES (?, ?)''',
                       (bioguide_id, name))

        # Insert data into the CongressMembers table
        cursor.execute('''INSERT INTO CongressMembers (CongressNumber, BioguideID)
                        VALUES (?, ?)''',
                       (congress_number, bioguide_id))

        line_count += 1

print(f'Processed {line_count} entries from legislators-current.json.')
conn.commit()

# Insert data from legislators-historical.json into Members and CongressMembers tables
with open('../data/vote/legislators-historical.json') as json_file:
    data = json.load(json_file)
    line_count = 0
    for row in data:
        bioguide_id = row["id"]["bioguide"]
        first_name = row["name"]["first"]
        last_name = row["name"]["last"]
        name = f"{first_name} {last_name}"
        congress_number = convert_year_to_congress(int(row["terms"][0]["start"][:4]))  # Assuming the first term is used for congress number

        # Insert data into the Members table
        cursor.execute('''INSERT INTO Members (BioguideID, Name)
                        VALUES (?, ?)''',
                       (bioguide_id, name))

        # Insert data into the CongressMembers table
        cursor.execute('''INSERT INTO CongressMembers (CongressNumber, BioguideID)
                        VALUES (?, ?)''',
                       (congress_number, bioguide_id))

        line_count += 1

print(f'Processed {line_count} entries from legislators-historical.json.')
conn.commit()

# Insert Congress numbers from 75 to 118
for congress_number in range(75, 119):
    cursor.execute("INSERT INTO Congress (CongressNumber) VALUES (?)", (congress_number,))

conn.commit()

# Close the database connection
conn.close()

# Create and insert into Bills table
# Function to parse XML file and extract required data
def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    bill_congress = None
    bill_type = None
    bill_number = None
    bill_description = None
    bill_summary = None

    # Retrieve data from XML
    for bill in root.iter('bill'):
        bill_congress_element = bill.find('congress')
        bill_congress = bill_congress_element.text if bill_congress_element is not None else None

        bill_type_element = bill.find('type')
        bill_type = bill_type_element.text.upper() if bill_type_element is not None else None  # Convert bill_type to uppercase

        bill_number_element = bill.find('number')
        bill_number = bill_number_element.text if bill_number_element is not None else None

        title_element = bill.find('title')
        bill_description = title_element.text if title_element is not None else None

        # Check if the 'summaries/summary/text' element exists
        summary_element = bill.find('summaries/summary/text')
        if summary_element is not None:
            bill_summary = clean_summary(summary_element.text)

    return bill_congress, bill_type, bill_number, bill_description, bill_summary

def parse_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        bill_congress = data.get('congress', -1)
        bill_type = data.get('bill_type', 'NULL').upper()
        bill_number = data.get('number', -1)
        bill_description = data.get('summary', {}).get('text', 'NULL')
        bill_summary = clean_summary(data.get('official_title', 'NULL'))

        return bill_congress, bill_type, bill_number, bill_description, bill_summary

# Function to clean the summary text and remove HTML tags and unwanted characters
def clean_summary(summary_text):
    # Remove HTML tags
    cleaned_text = re.sub(r'<.*?>', '', summary_text)
    # Remove newlines, tabs, and extra whitespaces
    cleaned_text = re.sub(r'[\n\t]', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    # Remove leading and trailing whitespaces
    cleaned_text = cleaned_text.strip()

    return cleaned_text

# Function to process XML files in the specified directory
def process_xml_files(directory_path):
    with sqlite3.connect('bills_congress.db') as conn:
        cursor = conn.cursor()

        # Drop bills table if it exists
        cursor.execute("DROP TABLE IF EXISTS bills")

        # Create bills table with auto-incrementing bill_id starting from 1
        cursor.execute('''CREATE TABLE bills (
                        BillID INTEGER PRIMARY KEY AUTOINCREMENT,
                        BillCongress INTEGER,
                        BillType TEXT,
                        BillNumber INTEGER,
                        BillDescription TEXT,
                        BillSummary TEXT)''')

        # Iterate through the directory structure
        for congress_dir in os.scandir(directory_path):
            if not congress_dir.is_dir():
                continue
            
            for bills_dir in os.scandir(congress_dir.path):
                if not bills_dir.is_dir():
                    continue

                for bill_type_dir in os.scandir(bills_dir.path):
                    if not bill_type_dir.is_dir():
                        continue

                    for bill_number_dir in os.scandir(bill_type_dir.path):
                        if not bill_number_dir.is_dir():
                            continue

                        for xml_file in os.scandir(bill_number_dir.path):
                            if xml_file.name.endswith('.xml'):
                                # Parse XML and extract data
                                bill_data = parse_xml(xml_file.path)
                            elif xml_file.name.endswith('.json'):
                                # Parse JSON and extract data
                                bill_data = parse_json(xml_file.path)
                            else:
                                continue

                            # Insert data into the bills table
                            cursor.execute('''INSERT INTO bills (
                                            BillCongress, BillType, BillNumber, BillSummary, BillDescription)
                                            VALUES (?, ?, ?, ?, ?)''', bill_data)

        conn.commit()

    # Close the database connection
    conn.close()

# Specify the directory path containing the XML files
# Note: you will have to download the scraped voting records data using the instructions in the README
data_directory = '../data/vote/bills/data'

# Process the XML files and populate the bills table
process_xml_files(data_directory)


# Creating and inserting into Rollcalls table
# Connect to the database
conn = sqlite3.connect('bills_congress.db')
cursor = conn.cursor()

cursor.execute('DROP TABLE IF EXISTS Rollcalls')

# Create the Rollcalls table
cursor.execute('''CREATE TABLE IF NOT EXISTS Rollcalls (
                    RollcallID TEXT NOT NULL,
                    MemberID TEXT,
                    VoteType TEXT,
                    BillID INTEGER,
                    PRIMARY KEY (RollcallID, MemberID),
                    FOREIGN KEY (MemberID) REFERENCES Members(BioguideID),
                    FOREIGN KEY (BillID) REFERENCES Bills(BillID)
                )''')

# Define the path to the JSON files
base_path = './data/'

# Iterate through the JSON files
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith('.json'):
            file_path = os.path.join(root, file)

            # Load the JSON data
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)

                # Extract the relevant information
                rollcall_id = data['vote_id']

                if "bill" not in data:
                    continue


                bill_type = data['bill']["type"].upper()  # Convert bill_type to uppercase
                bill_number = data['bill']["number"]

                # if bill_number == 117:
                #     assert 1 == 0

                # Get the bill_id from the "Bills" table based on billtype and billnumber and billcongress
                cursor.execute("SELECT BillID FROM Bills WHERE UPPER(BillType)=? AND BillNumber=? AND BillCongress=?" , (bill_type, int(bill_number), data['congress']))
                result = cursor.fetchone()
                if result:
                    bill_id = result[0]
                    print(f"Found bill_id={bill_id} for bill_type={bill_type}, bill_number={bill_number}, congress={data['congress']}")
                else:
                    bill_id = None
                    print(f"No matching bill_id found for bill_type={bill_type}, bill_number={bill_number}, congress={data['congress']}")

                # Get vote result for each member
                vote_data = []
                for vote_type in data["votes"]:
                    for member in data["votes"][vote_type]:
                        if not isinstance(member, dict):
                            continue
                        member_id = member["id"]

                        # Create query
                        vote_data.append((rollcall_id, member_id, vote_type,  bill_id))

                # Insert vote data into database
                query = "INSERT INTO Rollcalls (RollcallID, MemberID, VoteType, BillID) VALUES (?, ?, ?, ?)"

                # Execute query for each vote
                cursor.executemany(query, vote_data)

# Commit the changes and close the connection
conn.commit()
conn.close()
