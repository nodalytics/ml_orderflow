import os
import subprocess
import signal
import sys
import time
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from trend_analysis.utils.config import settings
from trend_analysis.utils.initializer import logger_instance

logger = logger_instance.get_logger()

class RobustScheduler:
    def __init__(self):
        self.scheduler = BlockingScheduler()
        self.is_running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)

    def _handle_exit(self, signum, frame):
        logger.info(f"Received signal {signum}. Stopping scheduler gracefully...")
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
        sys.exit(0)

    def run_pipeline(self):
        """
        Executes the DVC pipeline. max_instances=1 prevents overlaps.
        """
        if self.is_running:
            logger.warning("Pipeline is already running. Skipping this instance to avoid overlap.")
            return

        self.is_running = True
        start_time = datetime.now()
        logger.info(f"--- [START] Scheduled Pipeline Run at {start_time.strftime('%Y-%m-%d %H:%M:%S')} ---")
        
        try:
            # We use shell=True on Windows to ensure 'uv' is found correctly in the environment
            # and to handle command string parsing consistently.
            process = subprocess.run(
                "uv run dvc repro",
                shell=True,
                capture_output=True,
                text=True
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if process.returncode == 0:
                logger.info(f"Pipeline execution SUCCESSFUL. Duration: {duration:.2f}s")
                if process.stdout.strip():
                    logger.debug(f"DVC Output:\n{process.stdout}")
            else:
                logger.error(f"Pipeline execution FAILED with exit code {process.returncode}")
                logger.error(f"Error Output:\n{process.stderr}")
                
        except Exception as e:
            logger.exception(f"Unexpected error during pipeline execution: {e}")
        finally:
            self.is_running = False
            logger.info("--- [FINISH] Scheduled Pipeline Run ---")

    def start(self):
        schedule_params = settings.params.get('schedule', {})
        if not schedule_params.get('enabled', False):
            logger.warning("Scheduler is disabled in params.yaml. Exiting.")
            return

        interval_minutes = schedule_params.get('interval_minutes', 60)
        misfire_grace_time = schedule_params.get('misfire_grace_time', 60)
        logger.info(f"Robust Scheduler initialized. Interval: {interval_minutes} minutes, Misfire Grace: {misfire_grace_time}s.")

        # Add job with concurrency control (max_instances=1)
        # misfire_grace_time allows some delay if the system is busy
        self.scheduler.add_job(
            self.run_pipeline, 
            'interval', 
            minutes=interval_minutes,
            max_instances=1,
            misfire_grace_time=misfire_grace_time,
            next_run_time=datetime.now() # Run immediately on start
        )

        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            pass
        except Exception as e:
            logger.error(f"Scheduler stopped due to error: {e}")

if __name__ == "__main__":
    robust_scheduler = RobustScheduler()
    robust_scheduler.start()
