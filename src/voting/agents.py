import sqlite3
from nicknames import NickNamer
from utils.db_handler import query_database

president_nicknames = {
    "American Cincinnatus": "George Washington",
    "American Fabius": "George Washington",
    "Father of His Country": "George Washington",
    "His Excellency": "George Washington",
    "Sage of Mount Vernon": "George Washington",
    "Colossus of Independence": "John Adams",
    "Duke of Braintree": "John Adams",
    "Father of American Independence": "John Adams",
    "His Rotundity": "John Adams",
    "Old Sink or Swim": "John Adams",
    "Apostle of Democracy": "Thomas Jefferson",
    "Father of the Declaration of Independence": "Thomas Jefferson",
    "Long Tom": "Thomas Jefferson",
    "Man of the People": "Thomas Jefferson",
    "Red Fox": "Thomas Jefferson",
    "Sage of Monticello": "Thomas Jefferson",
    "Father of the Constitution": "James Madison",
    "Little Jemmy": "James Madison",
    "Era of Good Feelings President": "James Monroe",
    "Last Cocked Hat": "James Monroe",
    "Abolitionist": "John Quincy Adams",
    "Old Man Eloquent": "John Quincy Adams",
    "Mad Old Man From Massachusetts": "John Quincy Adams",
    "Andy": "Andrew Jackson",
    "Hero of New Orleans": "Andrew Jackson",
    "Jackass": "Andrew Jackson",
    "King Andrew": "Andrew Jackson",
    "King Mob": "Andrew Jackson",
    "Old Hickory": "Andrew Jackson",
    "People's President": "Andrew Jackson",
    "Sharp Knife": "Andrew Jackson",
    "American Talleyrand": "Martin Van Buren",
    "Blue Whiskey Van": "Martin Van Buren",
    "Careful Dutchman": "Martin Van Buren",
    "Dandy President": "Martin Van Buren",
    "Enchanter": "Martin Van Buren",
    "Great Manager": "Martin Van Buren",
    "Little Magician": "Martin Van Buren",
    "Machiavellian Bellshazzar": "Martin Van Buren",
    "Martin Van Ruin": "Martin Van Buren",
    "Master Spirit": "Martin Van Buren",
    "Matty Van": "Martin Van Buren",
    "Mistletoe Politician": "Martin Van Buren",
    "Old Kinderhook (OK)": "Martin Van Buren",
    "Red Fox of Kinderhook": "Martin Van Buren",
    "Sly Fox": "Martin Van Buren",
    "General Mum": "William Henry Harrison",
    "Old Granny": "William Henry Harrison",
    "Tippecanoe": "William Henry Harrison",
    "Old Tippecanoe": "William Henry Harrison",
    "Washington of the West": "William Henry Harrison",
    "His Accidency": "John Tyler",
    "First Dark Horse President": "James K. Polk",
    "Napoleon of the Stump": "James K. Polk",
    "Young Hickory": "James K. Polk",
    "Old Rough and Ready": "Zachary Taylor",
    "American Louis Philippe": "Millard Fillmore",
    "Last of the Whigs": "Millard Fillmore",
    "Wool Carder President": "Millard Fillmore",
    "Handsome Frank": "Franklin Pierce",
    "Purse": "Franklin Pierce",
    "Young Hickory of the Granite Hills": "Franklin Pierce",
    "Bachelor President": "James Buchanan",
    "Old Buck": "James Buchanan",
    "Old Public Functionary": "James Buchanan",
    "Ten-Cent Jimmy": "James Buchanan",
    "Abe": "Abraham Lincoln",
    "Honest Abe": "Abraham Lincoln",
    "Uncle Abe": "Abraham Lincoln",
    "Ancient One": "Abraham Lincoln",
    "Grand Wrestler": "Abraham Lincoln",
    "Great Emancipator": "Abraham Lincoln",
    "Rail-Splitter": "Abraham Lincoln",
    "Tycoon": "Abraham Lincoln",
    "Sir Veto": "Andrew Johnson",
    "Tennessee Tailor": "Andrew Johnson",
    "Butcher": "Ulysses S. Grant",
    "Great Hammerer": "Ulysses S. Grant",
    "Little Beauty": "Ulysses S. Grant",
    "Ulyss": "Ulysses S. Grant",
    "U.S. Grant": "Ulysses S. Grant",
    "Dark Horse President": "Rutherford B. Hayes",
    "His Fraudulency": "Rutherford B. Hayes",
    "Rutherfraud": "Rutherford B. Hayes",
    "Rud": "Rutherford B. Hayes",
    "Boatman Jim": "James Garfield",
    "Canal Boy": "James Garfield",
    "Preacher President": "James Garfield",
    "Chet": "Chester A. Arthur",
    "Elegant Arthur": "Chester A. Arthur",
    "Gentleman Boss": "Chester A. Arthur",
    "Prince Arthur": "Chester A. Arthur",
    "Dude President": "Chester A. Arthur",
    "Walrus": "Chester A. Arthur",
    "Big Steve": "Grover Cleveland",
    "Grover the Good": "Grover Cleveland",
    "His Obstinacy": "Grover Cleveland",
    "Uncle Jumbo": "Grover Cleveland",
    "Front Porch Campaigner": "Benjamin Harrison",
    "Grandfather's Hat": "Benjamin Harrison",
    "Human Iceberg": "Benjamin Harrison",
    "Kid Gloves Harrison": "Benjamin Harrison",
    "Little Ben": "Benjamin Harrison",
    "Pious Moonlight Dude": "Benjamin Harrison",
    "Idol of Ohio": "William McKinley",
    "Major": "William McKinley",
    "Napoleon of Protection": "William McKinley",
    "Wobbly Willie": "William McKinley",
    "Colonel": "Theodore Roosevelt",
    "Lion": "Theodore Roosevelt",
    "Teddy": "Theodore Roosevelt",
    "Telescope Teddy": "Theodore Roosevelt",
    "Teedie": "Theodore Roosevelt",
    "TR": "Theodore Roosevelt",
    "Trust Buster": "Theodore Roosevelt",
    "Big Bill": "William Howard Taft",
    "Big Chief": "William Howard Taft",
    "Big Lub": "William Howard Taft",
    "Sleeping Beauty": "William Howard Taft",
    "Coiner of Weasel Words": "Woodrow Wilson",
    "Phrasemaker": "Woodrow Wilson",
    "Professor": "Woodrow Wilson",
    "Schoolmaster": "Woodrow Wilson",
    "Wobbly Warren": "Warren G. Harding",
    "Cal": "Calvin Coolidge",
    "Cautious Cal": "Calvin Coolidge",
    "Cool Cal": "Calvin Coolidge",
    "Silent Cal": "Calvin Coolidge",
    "Chief": "Herbert Hoover",
    "Great Engineer": "Herbert Hoover",
    "Great Humanitarian": "Herbert Hoover",
    "FDR": "Franklin D. Roosevelt",
    "Feather-duster": "Franklin D. Roosevelt",
    "Sphinx": "Franklin D. Roosevelt",
    "That Man in the White House": "Franklin D. Roosevelt",
    "Give 'Em Hell Harry": "Harry S. Truman",
    "Haberdasher Harry": "Harry S. Truman",
    "Man From Independence": "Harry S. Truman",
    "Senator From Pendergast": "Harry S. Truman",
    "Ike": "Dwight D. Eisenhower",
    "American Erlander": "John F. Kennedy",
    "Jack": "John F. Kennedy",
    "JFK": "John F. Kennedy",
    "Little Blue Boy": "John F. Kennedy",
    "Rat Face": "John F. Kennedy",
    "Bull Johnson": "Lyndon B. Johnson",
    "Landslide Lyndon": "Lyndon B. Johnson",
    "LBJ": "Lyndon B. Johnson",
    "Light-Bulb Lyndon": "Lyndon B. Johnson",
    "Tricky Dick": "Richard Nixon",
    "Jerry": "Gerald Ford",
    "Junie": "Gerald Ford",
    "Mr. Nice Guy": "Gerald Ford",
    "Hot": "Jimmy Carter",
    "Jimmy": "Jimmy Carter",
    "Jimmy Cardigan": "Jimmy Carter",
    "Peanut Farmer": "Jimmy Carter",
    "Dutch": "Ronald Reagan",
    "Gipper": "Ronald Reagan",
    "Great Communicator": "Ronald Reagan",
    "Teflon President": "Ronald Reagan",
    "41": "George H. W. Bush",
    "Little Pop": "George H. W. Bush",
    "Poppy": "George H. W. Bush",
    "Bill": "Bill Clinton",
    "Bubba": "Bill Clinton",
    "Comeback Kid": "Bill Clinton",
    "Slick Willie": "Bill Clinton",
    "43": "George W. Bush",
    "Dubya": "George W. Bush",
    "Shrub": "George W. Bush",
    "Barry": "Barack Obama",
    "Barry O'Bomber": "Barack Obama",
    "Nobama": "Barack Obama",
    "No Drama Obama": "Barack Obama",
    "45": "Donald Trump",
    "The Donald": "Donald Trump",
    "Mr. Drumpf": "Donald Trump",
    "Orange Man": "Donald Trump",
    "President Snowflake": "Donald Trump",
    "Snowflake-in-Chief": "Donald Trump",
    "Orange Turd": "Donald Trump",
    "Donald Von ShitzInPantz": "Donald Trump",
    "Don Snoreleone": "Donald Trump",
    "Amtrak Joe": "Joe Biden",
    "Brandon": "Joe Biden",
    "Scranton Joe": "Joe Biden",
    "Sleepy Joe": "Joe Biden"
}

trump_nicknames = {
    "Sloppy Steve": "Steve Bannon",
    "Basement Biden": "Joe Biden",
    "Beijing Biden": "Joe Biden",
    "Crooked Joe Biden": "Joe Biden",
    "Sleepy Joe": "Joe Biden",
    "Slow Joe": "Joe Biden",
    "Little Michael": "Michael Bloomberg",
    "Mini Mike Bloomberg": "Michael Bloomberg",
    "Gov. Jerry \"Moonbeam\" Brown": "Jerry Brown",
    "Low Energy Jeb": "Jeb Bush",
    "Alfred E. Neuman": "Pete Buttigieg",
    "Boot-Edge-Edge": "Pete Buttigieg",
    "Coco Chow": "Elaine Chao",
    "Sloppy Chris Christie": "Chris Christie",
    "Wild Bill": "Bill Clinton",
    "Crazy Hillary": "Hillary Clinton",
    "Crooked Hillary": "Hillary Clinton",
    "Lyin' Hillary": "Hillary Clinton",
    "Beautiful Hillary": "Hillary Clinton",
    "Leakin' James Comey": "James Comey",
    "Lyin' James Comey": "James Comey",
    "Shadey James Comey": "James Comey",
    "Slimeball James Comey": "James Comey",
    "Slippery James Comey": "James Comey",
    "Lyin' Ted": "Ted Cruz",
    "Rob": "Ron DeSantis",
    "Ron DeSanctimonious": "Ron DeSantis",
    "Ron DeSanctus": "Ron DeSantis",
    "Meatball Ron": "Ron DeSantis",
    "Tiny D": "Ron DeSantis",
    "Ditzy DeVos": "Betsy DeVos",
    "Jeff Flakey": "Jeff Flake",
    "Birdbrain": "Nikki Haley",
    "Nimbra": "Nikki Haley",
    "Tricky Nikki": "Nikki Haley",
    "Aida Hutchinson": "Asa Hutchinson",
    "Peekaboo": "Letitia James",
    "Big Jim": "Jim Justice",
    "Mad Dog": "James Mattis",
    "Broken Old Crow": "Mitch McConnell",
    "Evan McMuffin": "Evan McMullin",
    "Wacky Omarosa": "Omarosa Manigault Newman",
    "Governor Newscum": "Gavin Newsom",
    "Evita": "Alexandria Ocasio-Cortez",
    "Crazy Nancy": "Nancy Pelosi",
    "Nervous Nancy": "Nancy Pelosi",
    "Wacky Jacky": "Jacky Rosen",
    "Mr. Peepers": "Rod Rosenstein",
    "Little Marco": "Marco Rubio",
    "Crazy Bernie": "Bernie Sanders",
    "Little Ben Sasse": "Ben Sasse",
    "Liddle' Adam Schiff": "Adam Schiff",
    "Pencil Neck": "Adam Schiff",
    "Shifty Schiff": "Adam Schiff",
    "Cryin' Chuck": "Chuck Schumer",
    "Mr. Magoo": "Jeff Sessions",
    "Deranged Jack Smith": "Jack Smith",
    "Big Luther": "Luther Strange",
    "Goofy Elizabeth Warren": "Elizabeth Warren",
    "Pocahontas": "Elizabeth Warren"
}


for key in trump_nicknames:
    president_nicknames[key] = trump_nicknames[key]

# lowercase all keys
president_nicknames = {k.lower(): v for k, v in president_nicknames.items()}

nn = NickNamer()

def lookup_agent(claim: str, fes: dict, db: sqlite3.Connection, nlp) -> str:
    """Given an agent, find the BioGuide ID of the agent in the database.

    Args:
        agent (str): string containing the name of the agent, taken from frame-semantic parser,
        db (sqlite3.Connection): database connection
        nlp (spacy.Language): Spacy NLP model for finding person names in agent

    Returns:
        str: BioGuide ID of the agent
    """

    agent = claim[fes["Agent"]["start"] : fes["Agent"]["end"]].strip()

    print(agent)
    candidate_ids = set()
    query = ""

    if agent.lower() in president_nicknames:
        actual_name = president_nicknames[agent.lower()]
        query = f"select bioguideId from members where upper(name) like upper('%{actual_name}%') or upper(wiki) like upper('%{actual_name}%')"

    else:
        agent_names = []

        # Get person names from agent
        for tok in nlp(agent):
            if tok.idx < fes["Agent"]["start"] or tok.idx > fes["Agent"]["end"]:
                continue
            
            if tok.ent_type_ == "PERSON":
                agent_names.append(tok.text)
        
        agent_name = " ".join(agent_names)

        if len(agent_names) == 0:
            query = f"select bioguideId from members where upper(name) like upper('%{agent}%') or upper(wiki) like upper('%{agent}%') order by bioguideId desc;"
        else:
            query = f"select bioguideId from members where upper(name) like upper('%{agent_name}%') or upper(wiki) like upper('%{agent_name}%') order by bioguideId desc;"

    print(query)
    results = query_database(db, query)

    for result in results:
        candidate_ids.add(result[0])

    if len(results) == 0:
        candidate_ids.add(None)

    return candidate_ids
