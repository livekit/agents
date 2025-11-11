# based on https://github.com/douglas/toxiproxy-python.

import socket
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import requests


class ProxyExists(Exception):
    pass


class NotFound(Exception):
    pass


class InvalidToxic(Exception):
    pass


def can_connect_to(host: str, port: int) -> bool:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        return sock.connect_ex((host, port)) == 0


def validate_response(response: requests.Response) -> requests.Response:
    if response.status_code == 409:
        raise ProxyExists(response.content)
    elif response.status_code == 404:
        raise NotFound(response.content)
    elif response.status_code == 400:
        raise InvalidToxic(response.content)
    return response


class APIConsumer:
    host: str = "toxiproxy"
    port: int = 8474

    @classmethod
    def get_base_url(cls) -> str:
        return f"http://{cls.host}:{cls.port}"

    @classmethod
    def get(cls, url: str, params: Optional[dict[str, Any]] = None, **kwargs) -> requests.Response:
        endpoint = cls.get_base_url() + url
        return validate_response(requests.get(url=endpoint, params=params, **kwargs))

    @classmethod
    def delete(cls, url: str, **kwargs) -> requests.Response:
        endpoint = cls.get_base_url() + url
        return validate_response(requests.delete(url=endpoint, **kwargs))

    @classmethod
    def post(cls, url: str, data: Any = None, json: Any = None, **kwargs) -> requests.Response:
        endpoint = cls.get_base_url() + url
        return validate_response(requests.post(url=endpoint, data=data, json=json, **kwargs))


@dataclass
class Toxic:
    type: str
    stream: str = "downstream"
    name: Optional[str] = None
    toxicity: float = 1.0
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = f"{self.type}_{self.stream}"


@dataclass
class Proxy:
    name: str
    upstream: str
    enabled: bool
    listen: str

    def __init__(self, name: str, upstream: str, enabled: bool, listen: str, **kwargs):
        self.name = name
        self.upstream = upstream
        self.enabled = enabled
        self.listen = listen

    @contextmanager
    def down(self) -> Iterator["Proxy"]:
        try:
            self.disable()
            yield self
        finally:
            self.enable()

    def toxics(self) -> dict[str, Toxic]:
        response = APIConsumer.get(f"/proxies/{self.name}/toxics")
        toxics_list = response.json()
        toxics_dict: dict[str, Toxic] = {}
        for toxic_data in toxics_list:
            toxic_data["proxy"] = self.name  # optionally add proxy info if needed elsewhere
            toxic_name = toxic_data.get(
                "name", f"{toxic_data.get('type')}_{toxic_data.get('stream', 'downstream')}"
            )
            toxics_dict[toxic_name] = Toxic(**toxic_data)
        return toxics_dict

    def get_toxic(self, toxic_name: str) -> Optional[Toxic]:
        return self.toxics().get(toxic_name)

    def add_toxic(
        self,
        *,
        type: str,
        stream: str = "downstream",
        name: Optional[str] = None,
        toxicity: float = 1.0,
        attributes: Optional[dict[str, Any]] = None,
    ) -> None:
        if name is None:
            name = f"{type}_{stream}"
        if attributes is None:
            attributes = {}
        json_payload = {
            "name": name,
            "type": type,
            "stream": stream,
            "toxicity": toxicity,
            "attributes": attributes,
        }
        APIConsumer.post(f"/proxies/{self.name}/toxics", json=json_payload).json()

    def destroy_toxic(self, toxic_name: str) -> bool:
        delete_url = f"/proxies/{self.name}/toxics/{toxic_name}"
        response = APIConsumer.delete(delete_url)
        return response.ok

    def destroy(self) -> bool:
        return APIConsumer.delete(f"/proxies/{self.name}").ok

    def disable(self) -> None:
        self.__enable_proxy(False)

    def enable(self) -> None:
        self.__enable_proxy(True)

    def __enable_proxy(self, enabled: bool) -> None:
        json_payload = {"enabled": enabled}
        APIConsumer.post(f"/proxies/{self.name}", json=json_payload).json()
        self.enabled = enabled


class Toxiproxy:
    def proxies(self) -> dict[str, Proxy]:
        response = APIConsumer.get("/proxies")
        proxies_data = response.json()
        proxies_dict: dict[str, Proxy] = {}
        for name, data in proxies_data.items():
            proxies_dict[name] = Proxy(**data)
        return proxies_dict

    def destroy_all(self) -> None:
        for proxy in list(self.proxies().values()):
            self.destroy(proxy)

    def get_proxy(self, proxy_name: str) -> Optional[Proxy]:
        return self.proxies().get(proxy_name)

    def running(self) -> bool:
        return can_connect_to(APIConsumer.host, APIConsumer.port)

    def version(self) -> Optional[bytes]:
        if self.running():
            return APIConsumer.get("/version").content
        return None

    def reset(self) -> bool:
        response = APIConsumer.post("/reset")
        return response.ok

    def create(
        self, upstream: str, name: str, listen: Optional[str] = None, enabled: Optional[bool] = None
    ) -> Proxy:
        if name in self.proxies():
            raise ProxyExists("This proxy already exists.")

        listen_addr = listen or "127.0.0.1:0"
        json_payload: dict = {"upstream": upstream, "name": name, "listen": listen_addr}
        if enabled is not None:
            json_payload["enabled"] = enabled

        proxy_info = APIConsumer.post("/proxies", json=json_payload).json()
        print(proxy_info)
        return Proxy(**proxy_info)

    def destroy(self, proxy: Proxy) -> bool:
        return proxy.destroy()

    def populate(self, proxies: list[dict[str, Any]]) -> list[Proxy]:
        populated_proxies: list[Proxy] = []
        for proxy_conf in proxies:
            name = proxy_conf["name"]
            existing = self.get_proxy(name)
            # If an existing proxy is found and its configuration differs, destroy it first.
            if existing and (
                existing.upstream != proxy_conf["upstream"]
                or existing.listen != proxy_conf["listen"]
            ):
                self.destroy(existing)
                existing = None
            if existing is None:
                proxy_instance = self.create(**proxy_conf)
                populated_proxies.append(proxy_instance)
        return populated_proxies

    def update_api_consumer(self, host: str, port: int) -> None:
        APIConsumer.host = host
        APIConsumer.port = port
