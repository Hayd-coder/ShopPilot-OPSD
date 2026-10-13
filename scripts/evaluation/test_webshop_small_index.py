"""Measure whether a Lucene index returns items from the 1k catalogue."""

import argparse
import json
from pathlib import Path

from pyserini.search.lucene import LuceneSearcher


parser = argparse.ArgumentParser()
parser.add_argument("--index", default="indexes_1k")
parser.add_argument("--expect-contained", action="store_true")
args = parser.parse_args()

repo_root = Path(__file__).resolve().parents[2]
webshop_root = (
    repo_root
    / "agent_system/environments/env_package/webshop/webshop"
)
with (webshop_root / "data/items_shuffle_1000.json").open(
    encoding="utf-8",
) as handle:
    products = json.load(handle)

catalogue_asins = {str(product["asin"]) for product in products}
searcher = LuceneSearcher(str(webshop_root / "search_engine" / args.index))
queries = [
    str(product.get("query") or product.get("name") or "")
    for product in products[:50]
]
hit_asins = [
    json.loads(searcher.doc(hit.docid).raw())["id"]
    for query in queries
    for hit in searcher.search(query, k=10)
]
outside_catalogue = [asin for asin in hit_asins if asin not in catalogue_asins]

print(
    f"queries={len(queries)} hits={len(hit_asins)} "
    f"unique_hits={len(set(hit_asins))} "
    f"outside_catalogue={len(outside_catalogue)}",
)
assert hit_asins, "The 1k Lucene index returned no results"
if args.expect_contained:
    assert not outside_catalogue, outside_catalogue[:10]
    print("WebShop 1k catalogue/index consistency: PASS")
