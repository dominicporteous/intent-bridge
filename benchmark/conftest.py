"""Process setup for executable benchmark runs."""

from dotenv import load_dotenv

# Application settings are immutable after their first import. Load repository
# configuration before pytest imports benchmark modules and their adapters.
load_dotenv()
