import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime, timezone

def setup_logging(level=logging.INFO):
    """
    Setup logging with console and file output.
    File logs are named by UTC date and rotate daily at midnight UTC.
    """
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Log file named by UTC date: app_2026-09-21.log
    utc_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = log_dir / f"app_{utc_date}.log"
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers (if any)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Log format with UTC timestamp
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S UTC"
    )
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with daily rotation at midnight UTC
    timed_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",  # rotate at midnight UTC
        interval=1,  # every 1 day
        backupCount=30,  # keep 30 days of logs
        encoding="utf-8",
        utc=True  # use UTC for rotation
    )
    
    # Set custom naming for rotated files
    # Default format: app_2026-09-21.log.2026-09-20
    # We want: app_2026-09-20.log
    def namer(name):
        """Convert backup filename to date-based format."""
        # name format from TimedRotatingFileHandler: app_2026-09-21.log.2026-09-20
        # We want to extract the date part and create: app_2026-09-20.log
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            return f"{parts[0].rsplit('_', 1)[0]}_{parts[1]}.log"
        return name
    
    timed_handler.namer = namer
    timed_handler.setLevel(level)
    timed_handler.setFormatter(formatter)
    root_logger.addHandler(timed_handler)
    
    # Log startup
    root_logger.info(f"Logging initialized | log_file={log_file} | level={logging.getLevelName(level)}")