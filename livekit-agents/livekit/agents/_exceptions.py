class AssignmentTimeoutError(Exception):
    """Raised when accepting a job but not receiving an assignment within the specified timeout.
    The server may have chosen another worker to handle this job."""

    pass
