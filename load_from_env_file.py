from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("WANDB_API_KEY")

# Install the datasets package
os.system("pip install datasets")
