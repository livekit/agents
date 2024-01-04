from livekit import agents
from livekit.plugins import silero, openai
import logging


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def job_request_cb(job_request: agents.JobRequest):
        logging.info("Received job request: %s", job_request)

    worker = agents.Worker(
        job_request_cb=job_request_cb, worker_type=agents.JobType.JT_ROOM
    )
    agents.run_app(worker)
