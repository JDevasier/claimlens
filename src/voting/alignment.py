from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from typing import Tuple
from openai import OpenAI
import json
import os

# Load environment variables from .env file
load_dotenv()

# Fact verification prompt template
fact_verification_prompt = """
Given the following factual claim, bill summary, and vote on the bill, evaluate whether the content of the bill summary and the voting record align with the given claim. You may consider factors such as the main objectives of the bill and unintended or implicit consequences. Your task is to determine if the information provided in the bill summary and the voting record supports or refutes the given factual claim. Return your explanation and one of the following labels in JSON format.
Bill Summary:
{summary}
Vote: {vote_type}
Claim: {claim}
Labels:
Supports - The vote on this bill directly or indirectly supports the claim.
Refutes - The vote on this bill explicitly refutes the claim.
Inconclusive - The vote on this bill does not provide enough information to support or refute the claim.
Irrelevant - The vote on this bill is not relevant to the claim at all.
"""

def query_gpt(summary: str, vote_type: str, claim: str) -> Tuple[str, str]:
    """Queries the GPT model with a given summary, vote type, and claim."""
    api_key = os.getenv("OPENAI_API_KEY")
    org_id = os.getenv("OPENAI_ORG")

    client = OpenAI(api_key=api_key, organization=org_id)

    try:
        # Make API request to OpenAI's GPT model
        response = client.chat.completions.create(
            model="gpt-4o",  # Adjust model as needed
            messages=[
                {"role": "system", "content": "Follow the user input exactly."},
                {"role": "user", "content": fact_verification_prompt.format(
                    summary=summary, vote_type=vote_type, claim=claim)}
            ]
        )
        # print(response.choices[0].message)
        print(response.choices[0].message.content)
        # Parse response content
        response_content = response.choices[0].message.content
        if response_content.startswith("```json"):
            response_content = response_content[7:-3]
        response_json = json.loads(response_content)
    except Exception as e:
        print(f"Error: {e}")
        return "Error", "An error occurred while verifying the claim."

    # Extract label and explanation from response JSON
    alignment = response_json.get('Label', response_json.get('label', 'Error'))
    explanation = response_json.get('Explanation', response_json.get('explanation', 'No explanation provided.'))
    
    return alignment, explanation

def query_chatgpt_parallel(summaries: list, vote_types: list, claims: list) -> list:
    """Queries GPT model in parallel for multiple summaries, vote types, and claims."""
    print("Querying GPT for alignment between claims and bill summaries...")

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(query_gpt, summaries, vote_types, claims))

    return results
