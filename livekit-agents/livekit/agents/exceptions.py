class AssignmentTimeoutError(Exception):
    """Assignment timed out"""

    pass


class JobCancelledError(Exception):
    """Job is cancelled and should not be processed.
    This can happens if the job becomes invalid (e.g publisher disconnects before accepting)"""

    pass


class AvailabilityAnsweredError(Exception):
    "Job request is already answered" ""

    pass
