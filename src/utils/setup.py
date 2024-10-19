from openai import OpenAI
import os

def load_environment_variables():
    """Load environment variables from the .env file."""
    from dotenv import load_dotenv
    load_dotenv()
    
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    
    return {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "proj_id": os.getenv("OPENAI_PROJ"),
        "org_id": os.getenv("OPENAI_ORG"),
    }

def setup_openai_client(api_config):
    """Initialize the OpenAI client."""
    return OpenAI(organization=api_config["org_id"], project=api_config["proj_id"])
