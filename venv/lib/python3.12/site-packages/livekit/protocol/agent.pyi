from . import models as _models
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class JobType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    JT_ROOM: _ClassVar[JobType]
    JT_PUBLISHER: _ClassVar[JobType]

class WorkerStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WS_AVAILABLE: _ClassVar[WorkerStatus]
    WS_FULL: _ClassVar[WorkerStatus]

class JobStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    JS_UNKNOWN: _ClassVar[JobStatus]
    JS_SUCCESS: _ClassVar[JobStatus]
    JS_FAILED: _ClassVar[JobStatus]
JT_ROOM: JobType
JT_PUBLISHER: JobType
WS_AVAILABLE: WorkerStatus
WS_FULL: WorkerStatus
JS_UNKNOWN: JobStatus
JS_SUCCESS: JobStatus
JS_FAILED: JobStatus

class AgentInfo(_message.Message):
    __slots__ = ("id", "name", "version")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    version: str
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., version: _Optional[str] = ...) -> None: ...

class Job(_message.Message):
    __slots__ = ("id", "type", "room", "participant")
    ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    ROOM_FIELD_NUMBER: _ClassVar[int]
    PARTICIPANT_FIELD_NUMBER: _ClassVar[int]
    id: str
    type: JobType
    room: _models.Room
    participant: _models.ParticipantInfo
    def __init__(self, id: _Optional[str] = ..., type: _Optional[_Union[JobType, str]] = ..., room: _Optional[_Union[_models.Room, _Mapping]] = ..., participant: _Optional[_Union[_models.ParticipantInfo, _Mapping]] = ...) -> None: ...

class WorkerMessage(_message.Message):
    __slots__ = ("register", "availability", "status", "job_update")
    REGISTER_FIELD_NUMBER: _ClassVar[int]
    AVAILABILITY_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    JOB_UPDATE_FIELD_NUMBER: _ClassVar[int]
    register: RegisterWorkerRequest
    availability: AvailabilityResponse
    status: UpdateWorkerStatus
    job_update: JobStatusUpdate
    def __init__(self, register: _Optional[_Union[RegisterWorkerRequest, _Mapping]] = ..., availability: _Optional[_Union[AvailabilityResponse, _Mapping]] = ..., status: _Optional[_Union[UpdateWorkerStatus, _Mapping]] = ..., job_update: _Optional[_Union[JobStatusUpdate, _Mapping]] = ...) -> None: ...

class ServerMessage(_message.Message):
    __slots__ = ("register", "availability", "assignment")
    REGISTER_FIELD_NUMBER: _ClassVar[int]
    AVAILABILITY_FIELD_NUMBER: _ClassVar[int]
    ASSIGNMENT_FIELD_NUMBER: _ClassVar[int]
    register: RegisterWorkerResponse
    availability: AvailabilityRequest
    assignment: JobAssignment
    def __init__(self, register: _Optional[_Union[RegisterWorkerResponse, _Mapping]] = ..., availability: _Optional[_Union[AvailabilityRequest, _Mapping]] = ..., assignment: _Optional[_Union[JobAssignment, _Mapping]] = ...) -> None: ...

class RegisterWorkerRequest(_message.Message):
    __slots__ = ("type", "worker_id", "version", "name")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    type: JobType
    worker_id: str
    version: str
    name: str
    def __init__(self, type: _Optional[_Union[JobType, str]] = ..., worker_id: _Optional[str] = ..., version: _Optional[str] = ..., name: _Optional[str] = ...) -> None: ...

class RegisterWorkerResponse(_message.Message):
    __slots__ = ("worker_id", "server_version")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    SERVER_VERSION_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    server_version: str
    def __init__(self, worker_id: _Optional[str] = ..., server_version: _Optional[str] = ...) -> None: ...

class AvailabilityRequest(_message.Message):
    __slots__ = ("job",)
    JOB_FIELD_NUMBER: _ClassVar[int]
    job: Job
    def __init__(self, job: _Optional[_Union[Job, _Mapping]] = ...) -> None: ...

class AvailabilityResponse(_message.Message):
    __slots__ = ("job_id", "available")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    available: bool
    def __init__(self, job_id: _Optional[str] = ..., available: bool = ...) -> None: ...

class JobStatusUpdate(_message.Message):
    __slots__ = ("job_id", "status", "error", "user_data")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    USER_DATA_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    status: JobStatus
    error: str
    user_data: str
    def __init__(self, job_id: _Optional[str] = ..., status: _Optional[_Union[JobStatus, str]] = ..., error: _Optional[str] = ..., user_data: _Optional[str] = ...) -> None: ...

class JobAssignment(_message.Message):
    __slots__ = ("job",)
    JOB_FIELD_NUMBER: _ClassVar[int]
    job: Job
    def __init__(self, job: _Optional[_Union[Job, _Mapping]] = ...) -> None: ...

class UpdateWorkerStatus(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: WorkerStatus
    def __init__(self, status: _Optional[_Union[WorkerStatus, str]] = ...) -> None: ...
