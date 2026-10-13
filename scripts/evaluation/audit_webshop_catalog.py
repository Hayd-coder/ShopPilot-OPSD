#!/usr/bin/env python3
"""Audit that a WebShop catalogue, attributes, goals, and Lucene index agree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--attributes", type=Path, required=True)
    parser.add_argument("--human-goals", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--require-human-goal-coverage", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from pyserini.index.lucene import IndexReader
    from pyserini.search.lucene import LuceneSearcher

    products = json.loads(args.catalog.read_text(encoding="utf-8"))
    attributes = json.loads(args.attributes.read_text(encoding="utf-8"))
    human_goals = json.loads(args.human_goals.read_text(encoding="utf-8"))

    product_asins = [str(product["asin"]).lower() for product in products]
    catalog = set(product_asins)
    attr_asins = {str(asin).lower() for asin in attributes}
    goal_asins = {str(asin).lower() for asin in human_goals}

    searcher = LuceneSearcher(str(args.index))
    reader = IndexReader(str(args.index))
    num_docs = int(reader.stats()["documents"])
    index_asins: list[str] = []
    unreadable_docs: list[int] = []
    for docid in range(num_docs):
        external_id = reader.convert_internal_docid_to_collection_docid(docid)
        if external_id is None:
            unreadable_docs.append(docid)
            continue
        index_asins.append(str(external_id).lower())
    index_set = set(index_asins)

    def evaluate_queries(field: str, *, fallback_to_name: bool = True) -> dict:
        hit_count = 0
        target_in_top_k = 0
        returned_outside_catalog = 0
        no_hit_asins: list[str] = []
        target_missing_asins: list[str] = []
        skipped_empty_asins: list[str] = []
        for product in products:
            asin = str(product["asin"]).lower()
            query_value = product.get(field)
            if not query_value and fallback_to_name:
                query_value = product.get("name")
            query = str(query_value or "").strip()
            if not query:
                skipped_empty_asins.append(asin)
                continue
            hits = searcher.search(query, k=args.top_k)
            if not hits:
                no_hit_asins.append(asin)
                continue
            hit_count += len(hits)
            returned = {
                str(json.loads(searcher.doc(hit.docid).raw())["id"]).lower()
                for hit in hits
            }
            returned_outside_catalog += len(returned - catalog)
            if asin in returned:
                target_in_top_k += 1
            else:
                target_missing_asins.append(asin)
        return {
            "hit_count": hit_count,
            "target_in_top_k": target_in_top_k,
            "returned_outside_catalog": returned_outside_catalog,
            "no_hit_asins": no_hit_asins,
            "target_missing_asins": target_missing_asins,
            "skipped_empty_asins": skipped_empty_asins,
        }

    broad_query = evaluate_queries("query")
    exact_title = evaluate_queries("name", fallback_to_name=False)

    checks = {
        "catalog_has_1000_rows": len(products) == 1000,
        "catalog_asins_unique": len(catalog) == len(products),
        "attributes_exactly_match_catalog": attr_asins == catalog,
        "index_doc_count_matches_catalog": num_docs == len(products),
        "index_asins_unique": len(index_set) == len(index_asins),
        "index_exactly_matches_catalog": index_set == catalog,
        "all_broad_queries_return_hits": not broad_query["no_hit_asins"],
        "all_broad_query_hits_inside_catalog": broad_query["returned_outside_catalog"] == 0,
        "all_nonempty_exact_title_queries_return_hits": not exact_title["no_hit_asins"],
        "all_exact_title_hits_inside_catalog": exact_title["returned_outside_catalog"] == 0,
    }
    if args.require_human_goal_coverage:
        checks["human_goals_cover_catalog"] = catalog <= goal_asins

    result = {
        "files": {
            "catalog": str(args.catalog),
            "attributes": str(args.attributes),
            "human_goals": str(args.human_goals),
            "index": str(args.index.resolve()),
        },
        "sha256": {
            "catalog": sha256(args.catalog),
            "attributes": sha256(args.attributes),
            "human_goals": sha256(args.human_goals),
        },
        "counts": {
            "catalog_rows": len(products),
            "catalog_unique_asins": len(catalog),
            "attribute_asins": len(attr_asins),
            "human_goal_asins": len(goal_asins),
            "catalog_asins_with_human_goals": len(catalog & goal_asins),
            "lucene_documents": num_docs,
            "lucene_unique_asins": len(index_set),
            "catalog_asins_missing_from_attributes": len(catalog - attr_asins),
            "attribute_asins_outside_catalog": len(attr_asins - catalog),
            "catalog_asins_missing_from_index": len(catalog - index_set),
            "index_asins_outside_catalog": len(index_set - catalog),
            "broad_queries_tested": len(products),
            "broad_query_hits": broad_query["hit_count"],
            "broad_queries_with_target_in_top_k": broad_query["target_in_top_k"],
            "broad_queries_without_any_hit": len(broad_query["no_hit_asins"]),
            "broad_query_hits_outside_catalog": broad_query["returned_outside_catalog"],
            "exact_title_queries_tested": len(products) - len(exact_title["skipped_empty_asins"]),
            "products_with_empty_title": len(exact_title["skipped_empty_asins"]),
            "exact_title_query_hits": exact_title["hit_count"],
            "exact_title_queries_with_target_in_top_k": exact_title["target_in_top_k"],
            "exact_title_queries_without_any_hit": len(exact_title["no_hit_asins"]),
            "exact_title_query_hits_outside_catalog": exact_title["returned_outside_catalog"],
            "unreadable_index_documents": len(unreadable_docs),
        },
        "checks": checks,
        "examples": {
            "missing_attributes": sorted(catalog - attr_asins)[:10],
            "attributes_outside_catalog": sorted(attr_asins - catalog)[:10],
            "missing_from_index": sorted(catalog - index_set)[:10],
            "index_outside_catalog": sorted(index_set - catalog)[:10],
            "broad_queries_without_hits": broad_query["no_hit_asins"][:10],
            "broad_query_target_missing_from_top_k": broad_query["target_missing_asins"][:10],
            "exact_title_queries_without_hits": exact_title["no_hit_asins"][:10],
            "products_with_empty_title": exact_title["skipped_empty_asins"][:10],
            "exact_title_query_target_missing_from_top_k": exact_title["target_missing_asins"][:10],
        },
    }

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("Audit failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
