import re
from typing import Optional

from bs4 import BeautifulSoup


# Define the metadata extraction functions
def simple_extractor(html: str | BeautifulSoup) -> str:
    if isinstance(html, str):
        soup = BeautifulSoup(html, "lxml")
    elif isinstance(html, BeautifulSoup):
        soup = html
    else:
        raise ValueError("Input should be either BeautifulSoup object or an HTML string")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def metadata_extractor(meta: dict, soup: BeautifulSoup, title_suffix: Optional[str] = None) -> dict:
    title_element = soup.find("title")
    description_element = soup.find("meta", attrs={"name": "description"})
    html_element = soup.find("html")
    title = title_element.get_text() if title_element else ""
    if title_suffix is not None:
        title += title_suffix

    return {
        "source": meta["loc"],
        "title": title,
        "description": description_element.get("content", "") if description_element else "",
        "language": html_element.get("lang", "") if html_element else "",
        **meta,
    }


def restaurant_metadata_func(record: dict, metadata: dict) -> dict:
    metadata["namespace"] = record.get("namespace")
    metadata["owner_id"] = record.get("owner_id")
    metadata["doc_id"] = record.get("doc_id")
    metadata["city"] = record.get("city")
    metadata["location"] = record.get("location")
    metadata["status"] = record.get("status")
    metadata["service_types"] = record.get("service_types")

    return metadata


def product_metadata_func(record: dict, metadata: dict) -> dict:
    metadata["namespace"] = record.get("namespace")
    metadata["owner_id"] = record.get("owner_id")
    metadata["doc_id"] = record.get("doc_id")
    metadata["status"] = record.get("status")
    metadata["upsell_target"] = record.get("upsell_target")

    return metadata
