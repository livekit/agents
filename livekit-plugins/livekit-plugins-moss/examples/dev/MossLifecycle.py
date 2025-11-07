import asyncio
import os
import uuid

import dotenv

from livekit.plugins.moss import DocumentInfo, MossClient

dotenv.load_dotenv()


async def main() -> None:
    project_id = os.environ.get("MOSS_PROJECT_ID")
    project_key = os.environ.get("MOSS_PROJECT_KEY")

    if not project_id or not project_key:
        raise RuntimeError("Set MOSS_PROJECT_ID and MOSS_PROJECT_KEY before running this script.")

    index_name = f"demo-{uuid.uuid4().hex[:8]}"
    client = MossClient(project_id=project_id, project_key=project_key)

    docs = [
        DocumentInfo(id="doc1", text="Track an order by logging into your account."),
        DocumentInfo(id="doc2", text="Our return policy allows 30-day returns."),
    ]

    print(f"Creating index {index_name}...")
    await client.create_index(index_name, docs, model_id="moss-minilm")

    print("Loading index...")
    await client.load_index(index_name)

    print("Querying index...")
    result = await client.query(index_name, "How do I return an item?", top_k=3)
    for hit in result.docs:
        print(f"score={hit.score:.4f} id={hit.id} text={hit.text}")

    print("Adding a new document...")
    await client.add_documents(
        index_name,
        [DocumentInfo(id="doc3", text="Contact support to change your shipping address.")],
    )

    print("Listing indexes...")
    indexes = await client.list_indexes()
    print([idx.name for idx in indexes])

    print("Cleaning up...")
    await client.delete_index(index_name)


if __name__ == "__main__":
    asyncio.run(main())
