import argparse
import logging
import os
from utils.setup import setup_logging
from claimsql import ClaimSQL

def main():
    # Set up logging
    setup_logging()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process a claim and get related bills and voting alignment.")
    parser.add_argument('claim', type=str, help='The claim to be processed.')
    parser.add_argument('--bills_to_return', type=int, default=5, help='Number of bills to return (default: 5).')
    parser.add_argument('--db', type=str, default='../data/sample.db', help='Database file path (default: sample.db).')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (default: cpu).')
    parser.add_argument('--fsp_model', type=str, default='../models/vote-fsp', help='Frame parser model path (default: roberta-base).')
    parser.add_argument('--vote_file', type=str, default='../data/fsp/Vote.xml', help='Vote frame info file path (default: ../data/fsp/Vote.xml).')
    parser.add_argument('--embedder_model', type=str, default='msmarco-distilbert-base-tas-b', help='Embedder model path (default: msmarco-distilbert-base-tas-b).')

    args = parser.parse_args()

    # Check if the database file exists
    if not os.path.isfile(args.db):
        logging.error(f"Database file '{args.db}' not found.")
        return

    # Initialize Claim2SQL
    try:
        claim_processor = ClaimSQL(args, embedder_model=args.embedder_model)  # Replace None with actual model if needed
    except Exception as e:
        logging.error(f"Failed to initialize Claim2SQL: {e}")
        return

    # Process the claim
    result = claim_processor.process_claim(args.claim, args.bills_to_return)

    print(result)

    # Output the results
    print("\n--- Claim Processing Result ---")
    print(f"Claim: {result['claim']}")
    if result['bills']:
        for bill in result['bills']:
            print(f"Bill Summary: {bill['summary']}\nVote Type: {bill['vote_type']}\nAlignment: {bill['alignment']}\nExplanation: {bill['alignment_explanation']}")
            print('---')
    else:
        print("No relevant bills found.")

if __name__ == "__main__":
    main()
