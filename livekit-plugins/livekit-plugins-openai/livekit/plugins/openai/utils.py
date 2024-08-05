from typing import Awaitable, Callable, Union

AsyncAzureADTokenProvider = Callable[[], Union[str, Awaitable[str]]]
