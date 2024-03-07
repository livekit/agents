from livekit.agents import JobProcess


async def test_job_process():
    job = JobProcess()
    await job.run()
    await job.request_exit()
