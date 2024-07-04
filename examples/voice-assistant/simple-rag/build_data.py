import asyncio
import uuid
import aiohttp
import pickle
from tqdm import tqdm
from livekit.plugins import rag, openai
from livekit.agents import tokenize

raw_data = open("raw_data.txt", "r").read()


async def main() -> None:
    http_session = aiohttp.ClientSession()
    paragraphs = tokenize.basic.tokenize_paragraphs(raw_data)

    sentence_chunker = rag.SentenceChunker()
    indexed_paragraphs = {}
    embeddings_n = 0
    for paragraph in paragraphs:
        p_uuid = uuid.uuid4()
        chunks = sentence_chunker.chunk(text=paragraph)
        indexed_paragraphs[p_uuid] = {
            "text": paragraph,
            "chunks": chunks,  # use smaller chunks for searching a paragraph
        }
        embeddings_n += len(chunks)

    # from this blog https://openai.com/index/new-embedding-models-and-api-updates/
    # 512 seems to provide good MTEB score with text-embedding-3-small
    index_builder = rag.annoy.IndexBuilder(f=512, metric="euclidean")
    pbar = tqdm(total=embeddings_n)

    for data_uuid, paragraph_data in indexed_paragraphs.items():
        for chunk in paragraph_data["chunks"]:
            results = await openai.create_embeddings(
                input=chunk,
                model="text-embedding-3-small",
                dimensions=512,
                http_session=http_session,
            )

            index_builder.add_item(results[0].embedding, data_uuid)
            pbar.update()

    index_builder.build()
    index_builder.save("vdb_data")

    # save data with pickle
    with open("my_data.pkl", "wb") as f:
        pickle.dump(indexed_paragraphs, f)

    await http_session.close()


if __name__ == "__main__":
    asyncio.run(main())
