"""Products other than the Agents SDKs, one Markdown file each in `products/`: the first line is
the product's summary, the rest is what a lookup returns."""

from pathlib import Path
from typing import NamedTuple

from livekit.agents import ToolError, function_tool


class Product(NamedTuple):
    summary: str
    body: str


def load_products() -> dict[str, Product]:
    """Every product in `products/`, keyed by its file name without `.md`."""
    products: dict[str, Product] = {}
    for path in sorted(Path(__file__).with_name("products").glob("*.md")):
        summary, _, body = path.read_text(encoding="utf-8").partition("\n")
        products[path.stem] = Product(summary.strip(), body.strip())
    return products


PRODUCTS = load_products()

LOOKUP_DESCRIPTION = (
    "Fetch the full knowledge base for one LiveKit product. Call this before answering any "
    "question about a LiveKit product other than the Agents SDKs; look it up rather than "
    "answering from memory. Products:\n"
    + "\n".join(f"- {name}: {product.summary}" for name, product in PRODUCTS.items())
)


@function_tool(description=LOOKUP_DESCRIPTION)
async def lookup_product(product: str) -> str:
    """
    Args:
        product: The product to fetch, one of the names listed above.
    """
    if product not in PRODUCTS:
        raise ToolError(f"unknown product {product!r}; valid products: {', '.join(PRODUCTS)}")
    return PRODUCTS[product].body
