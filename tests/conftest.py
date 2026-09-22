import os
import logging
from pathlib import Path

# Setup logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()]
)

# Set test environment
os.environ["LOG_LEVEL"] = "INFO"
TESTS_DIR = Path(__file__).parent
PROJECT_DIR = TESTS_DIR.parent