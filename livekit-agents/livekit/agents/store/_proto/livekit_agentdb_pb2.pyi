from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AgentDB(_message.Message):
    __slots__ = ()
    class CreateRequest(_message.Message):
        __slots__ = ("region", "ttl_seconds")
        REGION_FIELD_NUMBER: _ClassVar[int]
        TTL_SECONDS_FIELD_NUMBER: _ClassVar[int]
        region: str
        ttl_seconds: int
        def __init__(self, region: _Optional[str] = ..., ttl_seconds: _Optional[int] = ...) -> None: ...
    class CreateResponse(_message.Message):
        __slots__ = ("database_id", "expires_at")
        DATABASE_ID_FIELD_NUMBER: _ClassVar[int]
        EXPIRES_AT_FIELD_NUMBER: _ClassVar[int]
        database_id: str
        expires_at: int
        def __init__(self, database_id: _Optional[str] = ..., expires_at: _Optional[int] = ...) -> None: ...
    class GetRequest(_message.Message):
        __slots__ = ("database_id",)
        DATABASE_ID_FIELD_NUMBER: _ClassVar[int]
        database_id: str
        def __init__(self, database_id: _Optional[str] = ...) -> None: ...
    class ListRequest(_message.Message):
        __slots__ = ("page_size", "page_token")
        PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
        PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        page_size: int
        page_token: str
        def __init__(self, page_size: _Optional[int] = ..., page_token: _Optional[str] = ...) -> None: ...
    class ListResponse(_message.Message):
        __slots__ = ("databases", "next_page_token")
        DATABASES_FIELD_NUMBER: _ClassVar[int]
        NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
        databases: _containers.RepeatedCompositeFieldContainer[AgentDB.AgentDatabase]
        next_page_token: str
        def __init__(self, databases: _Optional[_Iterable[_Union[AgentDB.AgentDatabase, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
    class AgentDatabase(_message.Message):
        __slots__ = ("database_id", "region", "created_at", "expires_at", "tip")
        DATABASE_ID_FIELD_NUMBER: _ClassVar[int]
        REGION_FIELD_NUMBER: _ClassVar[int]
        CREATED_AT_FIELD_NUMBER: _ClassVar[int]
        EXPIRES_AT_FIELD_NUMBER: _ClassVar[int]
        TIP_FIELD_NUMBER: _ClassVar[int]
        database_id: str
        region: str
        created_at: int
        expires_at: int
        tip: int
        def __init__(self, database_id: _Optional[str] = ..., region: _Optional[str] = ..., created_at: _Optional[int] = ..., expires_at: _Optional[int] = ..., tip: _Optional[int] = ...) -> None: ...
    class DeleteRequest(_message.Message):
        __slots__ = ("database_id",)
        DATABASE_ID_FIELD_NUMBER: _ClassVar[int]
        database_id: str
        def __init__(self, database_id: _Optional[str] = ...) -> None: ...
    class DeleteResponse(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class DumpRequest(_message.Message):
        __slots__ = ("database_id",)
        DATABASE_ID_FIELD_NUMBER: _ClassVar[int]
        database_id: str
        def __init__(self, database_id: _Optional[str] = ...) -> None: ...
    class DumpResponse(_message.Message):
        __slots__ = ("download_url", "expires_at", "tip")
        DOWNLOAD_URL_FIELD_NUMBER: _ClassVar[int]
        EXPIRES_AT_FIELD_NUMBER: _ClassVar[int]
        TIP_FIELD_NUMBER: _ClassVar[int]
        download_url: str
        expires_at: int
        tip: int
        def __init__(self, download_url: _Optional[str] = ..., expires_at: _Optional[int] = ..., tip: _Optional[int] = ...) -> None: ...
    class Wire(_message.Message):
        __slots__ = ()
        class QueryLang(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            SQL: _ClassVar[AgentDB.Wire.QueryLang]
        SQL: AgentDB.Wire.QueryLang
        class ValueType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            NULL: _ClassVar[AgentDB.Wire.ValueType]
            INT: _ClassVar[AgentDB.Wire.ValueType]
            DOUBLE: _ClassVar[AgentDB.Wire.ValueType]
            TEXT: _ClassVar[AgentDB.Wire.ValueType]
            BLOB: _ClassVar[AgentDB.Wire.ValueType]
        NULL: AgentDB.Wire.ValueType
        INT: AgentDB.Wire.ValueType
        DOUBLE: AgentDB.Wire.ValueType
        TEXT: AgentDB.Wire.ValueType
        BLOB: AgentDB.Wire.ValueType
        class ClientMessage(_message.Message):
            __slots__ = ("request_id", "hello", "exec", "query", "batch", "begin", "commit", "rollback", "cancel", "credit", "ping")
            REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
            HELLO_FIELD_NUMBER: _ClassVar[int]
            EXEC_FIELD_NUMBER: _ClassVar[int]
            QUERY_FIELD_NUMBER: _ClassVar[int]
            BATCH_FIELD_NUMBER: _ClassVar[int]
            BEGIN_FIELD_NUMBER: _ClassVar[int]
            COMMIT_FIELD_NUMBER: _ClassVar[int]
            ROLLBACK_FIELD_NUMBER: _ClassVar[int]
            CANCEL_FIELD_NUMBER: _ClassVar[int]
            CREDIT_FIELD_NUMBER: _ClassVar[int]
            PING_FIELD_NUMBER: _ClassVar[int]
            request_id: int
            hello: AgentDB.Wire.Hello
            exec: AgentDB.Wire.Statement
            query: AgentDB.Wire.Statement
            batch: AgentDB.Wire.Batch
            begin: AgentDB.Wire.Begin
            commit: AgentDB.Wire.Commit
            rollback: AgentDB.Wire.Rollback
            cancel: AgentDB.Wire.Cancel
            credit: AgentDB.Wire.Credit
            ping: AgentDB.Wire.Ping
            def __init__(self, request_id: _Optional[int] = ..., hello: _Optional[_Union[AgentDB.Wire.Hello, _Mapping]] = ..., exec: _Optional[_Union[AgentDB.Wire.Statement, _Mapping]] = ..., query: _Optional[_Union[AgentDB.Wire.Statement, _Mapping]] = ..., batch: _Optional[_Union[AgentDB.Wire.Batch, _Mapping]] = ..., begin: _Optional[_Union[AgentDB.Wire.Begin, _Mapping]] = ..., commit: _Optional[_Union[AgentDB.Wire.Commit, _Mapping]] = ..., rollback: _Optional[_Union[AgentDB.Wire.Rollback, _Mapping]] = ..., cancel: _Optional[_Union[AgentDB.Wire.Cancel, _Mapping]] = ..., credit: _Optional[_Union[AgentDB.Wire.Credit, _Mapping]] = ..., ping: _Optional[_Union[AgentDB.Wire.Ping, _Mapping]] = ...) -> None: ...
        class ServerMessage(_message.Message):
            __slots__ = ("request_id", "hello_ok", "columns", "column_batch", "exec_result", "done", "error", "pong")
            REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
            HELLO_OK_FIELD_NUMBER: _ClassVar[int]
            COLUMNS_FIELD_NUMBER: _ClassVar[int]
            COLUMN_BATCH_FIELD_NUMBER: _ClassVar[int]
            EXEC_RESULT_FIELD_NUMBER: _ClassVar[int]
            DONE_FIELD_NUMBER: _ClassVar[int]
            ERROR_FIELD_NUMBER: _ClassVar[int]
            PONG_FIELD_NUMBER: _ClassVar[int]
            request_id: int
            hello_ok: AgentDB.Wire.HelloOk
            columns: AgentDB.Wire.Columns
            column_batch: AgentDB.Wire.ColumnBatch
            exec_result: AgentDB.Wire.ExecResult
            done: AgentDB.Wire.Done
            error: AgentDB.Wire.Error
            pong: AgentDB.Wire.Pong
            def __init__(self, request_id: _Optional[int] = ..., hello_ok: _Optional[_Union[AgentDB.Wire.HelloOk, _Mapping]] = ..., columns: _Optional[_Union[AgentDB.Wire.Columns, _Mapping]] = ..., column_batch: _Optional[_Union[AgentDB.Wire.ColumnBatch, _Mapping]] = ..., exec_result: _Optional[_Union[AgentDB.Wire.ExecResult, _Mapping]] = ..., done: _Optional[_Union[AgentDB.Wire.Done, _Mapping]] = ..., error: _Optional[_Union[AgentDB.Wire.Error, _Mapping]] = ..., pong: _Optional[_Union[AgentDB.Wire.Pong, _Mapping]] = ...) -> None: ...
        class Value(_message.Message):
            __slots__ = ("null_value", "int_value", "double_value", "text_value", "blob_value")
            NULL_VALUE_FIELD_NUMBER: _ClassVar[int]
            INT_VALUE_FIELD_NUMBER: _ClassVar[int]
            DOUBLE_VALUE_FIELD_NUMBER: _ClassVar[int]
            TEXT_VALUE_FIELD_NUMBER: _ClassVar[int]
            BLOB_VALUE_FIELD_NUMBER: _ClassVar[int]
            null_value: bool
            int_value: int
            double_value: float
            text_value: str
            blob_value: bytes
            def __init__(self, null_value: bool = ..., int_value: _Optional[int] = ..., double_value: _Optional[float] = ..., text_value: _Optional[str] = ..., blob_value: _Optional[bytes] = ...) -> None: ...
        class Statement(_message.Message):
            __slots__ = ("sql", "params", "lang")
            SQL_FIELD_NUMBER: _ClassVar[int]
            PARAMS_FIELD_NUMBER: _ClassVar[int]
            LANG_FIELD_NUMBER: _ClassVar[int]
            sql: str
            params: _containers.RepeatedCompositeFieldContainer[AgentDB.Wire.Value]
            lang: AgentDB.Wire.QueryLang
            def __init__(self, sql: _Optional[str] = ..., params: _Optional[_Iterable[_Union[AgentDB.Wire.Value, _Mapping]]] = ..., lang: _Optional[_Union[AgentDB.Wire.QueryLang, str]] = ...) -> None: ...
        class Batch(_message.Message):
            __slots__ = ("statements",)
            STATEMENTS_FIELD_NUMBER: _ClassVar[int]
            statements: _containers.RepeatedCompositeFieldContainer[AgentDB.Wire.Statement]
            def __init__(self, statements: _Optional[_Iterable[_Union[AgentDB.Wire.Statement, _Mapping]]] = ...) -> None: ...
        class Begin(_message.Message):
            __slots__ = ()
            def __init__(self) -> None: ...
        class Commit(_message.Message):
            __slots__ = ()
            def __init__(self) -> None: ...
        class Rollback(_message.Message):
            __slots__ = ()
            def __init__(self) -> None: ...
        class Cancel(_message.Message):
            __slots__ = ()
            def __init__(self) -> None: ...
        class Credit(_message.Message):
            __slots__ = ("batches",)
            BATCHES_FIELD_NUMBER: _ClassVar[int]
            batches: int
            def __init__(self, batches: _Optional[int] = ...) -> None: ...
        class Ping(_message.Message):
            __slots__ = ("timestamp_ms",)
            TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
            timestamp_ms: int
            def __init__(self, timestamp_ms: _Optional[int] = ...) -> None: ...
        class Pong(_message.Message):
            __slots__ = ("last_ping_timestamp_ms", "timestamp_ms")
            LAST_PING_TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
            TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
            last_ping_timestamp_ms: int
            timestamp_ms: int
            def __init__(self, last_ping_timestamp_ms: _Optional[int] = ..., timestamp_ms: _Optional[int] = ...) -> None: ...
        class Hello(_message.Message):
            __slots__ = ("token", "database_id")
            TOKEN_FIELD_NUMBER: _ClassVar[int]
            DATABASE_ID_FIELD_NUMBER: _ClassVar[int]
            token: str
            database_id: str
            def __init__(self, token: _Optional[str] = ..., database_id: _Optional[str] = ...) -> None: ...
        class HelloOk(_message.Message):
            __slots__ = ("tip", "ping_interval_ms", "ping_timeout_ms")
            TIP_FIELD_NUMBER: _ClassVar[int]
            PING_INTERVAL_MS_FIELD_NUMBER: _ClassVar[int]
            PING_TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
            tip: int
            ping_interval_ms: int
            ping_timeout_ms: int
            def __init__(self, tip: _Optional[int] = ..., ping_interval_ms: _Optional[int] = ..., ping_timeout_ms: _Optional[int] = ...) -> None: ...
        class Columns(_message.Message):
            __slots__ = ("names",)
            NAMES_FIELD_NUMBER: _ClassVar[int]
            names: _containers.RepeatedScalarFieldContainer[str]
            def __init__(self, names: _Optional[_Iterable[str]] = ...) -> None: ...
        class Column(_message.Message):
            __slots__ = ("types", "ints", "doubles", "text_data", "text_ends", "blob_data", "blob_ends")
            TYPES_FIELD_NUMBER: _ClassVar[int]
            INTS_FIELD_NUMBER: _ClassVar[int]
            DOUBLES_FIELD_NUMBER: _ClassVar[int]
            TEXT_DATA_FIELD_NUMBER: _ClassVar[int]
            TEXT_ENDS_FIELD_NUMBER: _ClassVar[int]
            BLOB_DATA_FIELD_NUMBER: _ClassVar[int]
            BLOB_ENDS_FIELD_NUMBER: _ClassVar[int]
            types: bytes
            ints: _containers.RepeatedScalarFieldContainer[int]
            doubles: _containers.RepeatedScalarFieldContainer[float]
            text_data: bytes
            text_ends: _containers.RepeatedScalarFieldContainer[int]
            blob_data: bytes
            blob_ends: _containers.RepeatedScalarFieldContainer[int]
            def __init__(self, types: _Optional[bytes] = ..., ints: _Optional[_Iterable[int]] = ..., doubles: _Optional[_Iterable[float]] = ..., text_data: _Optional[bytes] = ..., text_ends: _Optional[_Iterable[int]] = ..., blob_data: _Optional[bytes] = ..., blob_ends: _Optional[_Iterable[int]] = ...) -> None: ...
        class ColumnBatch(_message.Message):
            __slots__ = ("columns", "rows")
            COLUMNS_FIELD_NUMBER: _ClassVar[int]
            ROWS_FIELD_NUMBER: _ClassVar[int]
            columns: _containers.RepeatedCompositeFieldContainer[AgentDB.Wire.Column]
            rows: int
            def __init__(self, columns: _Optional[_Iterable[_Union[AgentDB.Wire.Column, _Mapping]]] = ..., rows: _Optional[int] = ...) -> None: ...
        class ExecResult(_message.Message):
            __slots__ = ("rows_affected", "last_insert_id", "tip")
            ROWS_AFFECTED_FIELD_NUMBER: _ClassVar[int]
            LAST_INSERT_ID_FIELD_NUMBER: _ClassVar[int]
            TIP_FIELD_NUMBER: _ClassVar[int]
            rows_affected: int
            last_insert_id: int
            tip: int
            def __init__(self, rows_affected: _Optional[int] = ..., last_insert_id: _Optional[int] = ..., tip: _Optional[int] = ...) -> None: ...
        class Done(_message.Message):
            __slots__ = ("tip", "total_rows")
            TIP_FIELD_NUMBER: _ClassVar[int]
            TOTAL_ROWS_FIELD_NUMBER: _ClassVar[int]
            tip: int
            total_rows: int
            def __init__(self, tip: _Optional[int] = ..., total_rows: _Optional[int] = ...) -> None: ...
        class Error(_message.Message):
            __slots__ = ("code", "message")
            CODE_FIELD_NUMBER: _ClassVar[int]
            MESSAGE_FIELD_NUMBER: _ClassVar[int]
            code: str
            message: str
            def __init__(self, code: _Optional[str] = ..., message: _Optional[str] = ...) -> None: ...
        def __init__(self) -> None: ...
    def __init__(self) -> None: ...
