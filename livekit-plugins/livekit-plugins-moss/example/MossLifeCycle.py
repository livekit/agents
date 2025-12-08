import asyncio
import logging
import os
import sys
import uuid

import dotenv

from livekit.plugins.moss import DocumentInfo, MossClient

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

dotenv.load_dotenv()


async def main() -> None:
    project_id = os.environ.get("MOSS_PROJECT_ID")
    project_key = os.environ.get("MOSS_PROJECT_KEY")

    if not project_id or not project_key:
        logger.error("MOSS_PROJECT_ID and MOSS_PROJECT_KEY must be set in environment")
        logger.error("Set MOSS_PROJECT_ID and MOSS_PROJECT_KEY before running this script.")
        sys.exit(1)

    index_name = f"demo-{uuid.uuid4().hex[:8]}"
    logger.info(f"Initializing MossClient with project_id={project_id[:8]}...")
    client = MossClient(project_id=project_id, project_key=project_key)
    logger.info("MossClient initialized successfully")

    docs = [
        DocumentInfo(id="doc1", text="Track an order by logging into your account."),
        DocumentInfo(id="doc2", text="Our return policy allows 30-day returns."),
        DocumentInfo(id="doc3", text="Contact support to change your shipping address."),
        DocumentInfo(id="doc4", text="Free shipping is available on orders over $50."),
        DocumentInfo(id="doc5", text="You can cancel your order within 24 hours of placement."),
        DocumentInfo(id="doc6", text="Refunds are processed within 5-7 business days after we receive your return."),
        DocumentInfo(id="doc7", text="Express shipping options are available at checkout for faster delivery."),
        DocumentInfo(id="doc8", text="Create an account to save your payment methods and shipping addresses."),
        DocumentInfo(id="doc9", text="Gift cards can be purchased online and used for any purchase."),
        DocumentInfo(id="doc10", text="Subscribe to our newsletter to receive exclusive discounts and updates."),
    ]
    logger.info(f"Prepared {len(docs)} documents for index creation")

    # ---------- Moss Create Index ----------
    logger.info(f"Creating index '{index_name}' with model 'moss-minilm'...")
    created = await client.create_index(index_name, docs, model_id="moss-minilm")
    logger.info(f"create_index() returned: {created}")
    if created:
        logger.info(f"Successfully created index '{index_name}'")
    else:
        logger.warning(f"Index creation returned False for '{index_name}'")

    # ---------- Moss Get Index ----------
    logger.info(f"Retrieving index information for '{index_name}'...")
    index_info = await client.get_index(index_name)
    logger.info(f"get_index() returned: {index_info}")
    logger.info(f"Index info - name: {index_info.name}, model: {getattr(index_info, 'model_id', 'N/A')}")

    # ---------- Moss Load Index ----------
    logger.info(f"Loading index '{index_name}' into memory...")
    load_result = await client.load_index(index_name)
    logger.info(f"load_index() returned: {load_result}")
    logger.info(f"Index '{index_name}' loaded successfully")

    # ---------- Moss Query ----------
    query_text = "How do I return an item?"
    logger.info(f"Querying index '{index_name}' with query: '{query_text}' (top_k=3)...")
    result = await client.query(index_name, query_text, top_k=3)
    logger.info(f"query() returned: SearchResult with {len(result.docs)} docs")
    logger.info(f"Query returned {len(result.docs)} results")
    for i, hit in enumerate(result.docs, 1):
        logger.info(f"  Result {i}: score={hit.score:.4f}, id={hit.id}, text={hit.text[:50]}...")

    # ---------- Moss Add Documents ----------
    new_doc = DocumentInfo(id="doc11", text="Live chat support is available Monday through Friday, 9 AM to 5 PM EST.")
    logger.info(f"Adding document '{new_doc.id}' to index '{index_name}'...")
    add_result = await client.add_documents(index_name, [new_doc], options=None)
    logger.info(f"add_documents() returned: {add_result}")
    logger.info(f"Add documents result: {add_result}")

    # ---------- Moss Get Documents ----------
    logger.info("Retrieving all documents from index...")
    all_docs = await client.get_docs(index_name, options=None)
    logger.info(f"get_docs() returned: list with {len(all_docs)} DocumentInfo objects")
    logger.info(f"Index '{index_name}' now contains {len(all_docs)} documents")
    for doc in all_docs:
        logger.info(f"  - Document ID: {doc.id}, text: {doc.text[:60]}...")

    logger.info("Listing all indexes in project...")
    indexes = await client.list_indexes()
    logger.info(f"list_indexes() returned: list with {len(indexes)} IndexInfo objects")
    logger.info(f"Found {len(indexes)} indexes in project")
    for idx in indexes:
        logger.info(f"  - {idx.name}")

    # ---------- Moss Delete Documents ----------
    logger.info(f"Deleting documents 'doc1' and 'doc2' from index '{index_name}'...")
    delete_result = await client.delete_docs(index_name, ["doc1", "doc2"])
    logger.info(f"delete_docs() returned: {delete_result}")
    logger.info(f"Delete documents result: {delete_result}")

    # ---------- Moss Delete Index ----------
    logger.info(f"Deleting index '{index_name}'...")
    deleted = await client.delete_index(index_name)
    logger.info(f"delete_index() returned: {deleted}")
    if deleted:
        logger.info(f"Successfully deleted index '{index_name}'")
    else:
        logger.warning(f"Index deletion returned False for '{index_name}'")

    logger.info("Moss lifecycle demo completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
