import logging

from livekit.agents import JobContext, JobRequest, WorkerOptions, cli


async def entrypoint(ctx: JobContext):
    logging.info("starting entrypoint")

    # Add your agent logic here!


async def request_fnc(req: JobRequest) -> None:
    logging.info("received request %s", req)
    await req.accept(entrypoint)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc))
