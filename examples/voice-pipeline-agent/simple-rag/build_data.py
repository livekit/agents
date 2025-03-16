import asyncio
import pickle
import uuid

import aiohttp
from tqdm import tqdm

from livekit.agents import tokenize
from livekit.plugins import openai, rag

# from this blog https://openai.com/index/new-embedding-models-and-api-updates/
# 512 seems to provide good MTEB score with text-embedding-3-small
embeddings_dimension = 1536
raw_data = open("raw_data.txt").read()


async def _create_embeddings(
    input: str, http_session: aiohttp.ClientSession
) -> openai.EmbeddingData:
    results = await openai.create_embeddings(
        input=[input],
        model="text-embedding-3-small",
        dimensions=embeddings_dimension,
        http_session=http_session,
    )
    return results[0]


async def main() -> None:
    async with aiohttp.ClientSession() as http_session:
        idx_builder = rag.annoy.IndexBuilder(f=embeddings_dimension, metric="angular")

        paragraphs_by_uuid = {}
        for p in tokenize.basic.tokenize_paragraphs(raw_data):
            p_uuid = uuid.uuid4()
            paragraphs_by_uuid[p_uuid] = p

        for p_uuid, paragraph in tqdm(paragraphs_by_uuid.items()):
            resp = await _create_embeddings(paragraph, http_session)
            idx_builder.add_item(resp.embedding, p_uuid)

        idx_builder.build()
        idx_builder.save("vdb_data")

        # save data with pickle
        with open("my_data.pkl", "wb") as f:
            pickle.dump(paragraphs_by_uuid, f)


if __name__ == "__main__":
    asyncio.run(main())
