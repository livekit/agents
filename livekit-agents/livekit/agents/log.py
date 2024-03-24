import logging

<<<<<<< Updated upstream
worker_logger = logging.getLogger("livekit.worker")
job_logger = logging.getLogger("livekit.job")

# Only used inside a job process
process_logger = logging.getLogger("livekit.internal.process")
=======
logger = logging.getLogger("livekit.agents")
>>>>>>> Stashed changes
