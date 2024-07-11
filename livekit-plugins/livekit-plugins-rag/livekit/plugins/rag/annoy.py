import pathlib
import pickle
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import annoy

# https://github.com/spotify/annoy

__all__ = ["AnnoyIndex", "IndexBuilder", "Item", "Metric"]

Metric = Literal["angular", "euclidean", "manhattan", "hamming", "dot"]

ANNOY_FILE = "index.annoy"
METADATA_FILE = "metadata.pkl"


@dataclass
class Item:
    i: int
    userdata: Any
    vector: list[float]


@dataclass
class _FileData:
    f: int
    metric: Metric
    userdata: dict[int, Any]


@dataclass
class QueryResult:
    userdata: Any
    distance: float


class AnnoyIndex:
    def __init__(self, index: annoy.AnnoyIndex, filedata: _FileData) -> None:
        self._index = index
        self._filedata = filedata

    @classmethod
    def load(cls, path: str) -> "AnnoyIndex":
        p = pathlib.Path(path)
        index_path = p / ANNOY_FILE
        metadata_path = p / METADATA_FILE

        with open(metadata_path, "rb") as f:
            metadata: _FileData = pickle.load(f)

        index = annoy.AnnoyIndex(metadata.f, metadata.metric)
        index.load(str(index_path))
        return cls(index, metadata)

    @property
    def size(self) -> int:
        return self._index.get_n_items()

    def items(self) -> Iterable[Item]:
        for i in range(self._index.get_n_items()):
            item = Item(
                i=i,
                userdata=self._filedata.userdata[i],
                vector=self._index.get_item_vector(i),
            )
            yield item

    def query(
        self, vector: list[float], n: int, search_k: int = -1
    ) -> list[QueryResult]:
        ids = self._index.get_nns_by_vector(
            vector, n, search_k=search_k, include_distances=True
        )
        return [
            QueryResult(userdata=self._filedata.userdata[i], distance=distance)
            for i, distance in zip(*ids)
        ]


class IndexBuilder:
    def __init__(self, f: int, metric: Metric) -> None:
        self._index = annoy.AnnoyIndex(f, metric)
        self._filedata = _FileData(f=f, metric=metric, userdata={})
        self._i = 0

    def save(self, path: str) -> None:
        p = pathlib.Path(path)
        p.mkdir(parents=True, exist_ok=True)
        index_path = p / ANNOY_FILE
        metadata_path = p / METADATA_FILE
        self._index.save(str(index_path))
        with open(metadata_path, "wb") as f:
            pickle.dump(self._filedata, f)

    def build(self, trees: int = 50, jobs: int = -1) -> AnnoyIndex:
        # n_jobs=-1 means use all available cores
        self._index.build(n_trees=trees, n_jobs=jobs)
        return AnnoyIndex(self._index, self._filedata)

    def add_item(self, vector: list[float], userdata: Any) -> None:
        self._index.add_item(self._i, vector)
        self._filedata.userdata[self._i] = userdata
        self._i += 1
