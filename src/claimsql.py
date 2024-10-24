import logging
import spacy
from utils.file_io import write_members_to_file
from utils.db_handler import connect_to_db, query_database
from utils.text_processing import clean_sentence
from voting.bills import lookup_bill
from voting.agents import lookup_agent
from voting.alignment import query_chatgpt_parallel
from models import BillFinder, FrameParser

class ClaimSQL:
    def __init__(self, args, embedder_model):
        """Initialize Claim2SQL with a frame parser, database, and bill search model."""
        self.fsp = FrameParser(args)
        self.db = connect_to_db(args.db)
        self.nlp = spacy.load("en_core_web_sm")
        self.bill_search_model = BillFinder(self.db, embedder_model, args)
        # self.build_member_txt()

    def build_member_txt(self):
        """Retrieve congressional members from DB and write to file."""
        members = query_database(self.db, "SELECT * from Members;")
        write_members_to_file(members)

    def process_claim(self, claim: str, bills_to_return: int = 5):
        """Main function to process a claim and return related bills and voting alignment."""
        claim_cleaned = clean_sentence(claim)
        parsed_claim = self.parse_claim(claim_cleaned)
        
        parsed_claim = parsed_claim[0] if isinstance(parsed_claim, list) else parsed_claim

        # claim_cleaned = "John Smith voted against better education."
        # parsed_claim = {'Agent': {'start': 0, 'end': len('John Smith')}, 
        #                 'Issue': {'start': len('John Smith voted against '), 'end': len(claim_cleaned)-1}}
        logging.info(parsed_claim)
        if not parsed_claim:
            return self.default_return(claim_cleaned)
        
        agent = self.lookup_agent(claim_cleaned, parsed_claim)
        logging.info(f"Agent: {agent}")
        if not agent:
            return self.default_return(claim_cleaned)
        
        issue_text = claim_cleaned[parsed_claim["Issue"]["start"]:parsed_claim["Issue"]["end"]].strip()
        bills = self.lookup_bills(issue_text, agent, bills_to_return)
        if not bills:
            return self.default_return(claim_cleaned)
        
        return self.get_bill_alignment(claim_cleaned, bills, agent)

    def parse_claim(self, claim):
        """Parse claim into frame elements using the FrameParser."""
        is_vote_frame, fes = self.fsp(claim)
        if not is_vote_frame:
            return None
        return fes

    def lookup_agent(self, claim, fes):
        """Look up the agent in the database."""
        try:
            agent = lookup_agent(claim, fes, self.db, self.nlp)
        except Exception as e:
            logging.error(f"Error finding agent: {e}")
            return None
        return agent.pop() if agent else None

    def lookup_bills(self, issue_fe, agent, bills_to_return):
        """Look up bills related to the issue and agent."""
        try:
            bills = lookup_bill(issue_fe, self.bill_search_model, agent, bills_to_return)
        except Exception as e:
            logging.error(f"Error finding bills: {e}")
            return None
        return bills

    def get_bill_alignment(self, claim, bills, agent):
        """Check bill alignment with the claim using the OpenAI API."""
        bill_details = self.retrieve_bill_details(bills, agent)
        bill_alignments = query_chatgpt_parallel(
            [bill["summary"] for bill in bill_details],
            [bill["vote_type"] for bill in bill_details],
            [claim for _ in bill_details]
        )
        for i, alignment in enumerate(bill_alignments):
            bill_details[i]["alignment"], bill_details[i]["alignment_explanation"] = alignment

        return {"claim": claim, "bills": bill_details}

    def retrieve_bill_details(self, bills, agent):
        """Retrieve detailed bill information including summaries and votes."""
        details = []
        for bill in bills:
            summary = query_database(self.db, f"SELECT BillSummary from bills WHERE BillID = ?", (bill,))
            vote_type = query_database(self.db, f"SELECT VoteType from Rollcalls WHERE BillID = ? AND MemberID = ?", (bill, agent))
            details.append({"summary": summary, "vote_type": vote_type, "alignment": None, "alignment_explanation": None})
        return details

    def default_return(self, claim):
        """Return a default response when no meaningful results are found."""
        return {"claim": claim, "bills": []}
