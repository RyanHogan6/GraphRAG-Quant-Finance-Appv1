"""
Deterministic JSON query plan to AQL converter.
Used by the two-step NL -> JSON -> AQL flow (replaces hard-coded AQL prompts).

Fixes applied from validation results: primary_collection alignment, duplicate
traversal skip, path-valid filters/sorts/aggregations/returns, COLLECT-with-total
safety, aggregation/sort as list or dict, and unique from_var for orphaned
traversals (avoids "variable assigned multiple times" e.g. eia_crude vs eia_natgas).
"""
from typing import Dict, Any, Optional, Callable


def json_to_aql(
    json_plan: Dict[str, Any],
    log: Optional[Callable[[str], None]] = None,
) -> str:
    """Convert JSON query plan to AQL. log() receives SKIP/AUTO-FIX messages (default: no-op)."""
    if not json_plan:
        return "// Invalid JSON plan"

    def out(msg: str) -> None:
        if log:
            log(msg)

    aql_parts = []
    variables = {}

    primary = json_plan.get("primary_collection", "Company")
    traversals = json_plan.get("traversals", [])

    if traversals and traversals[0].get("from_collection") != primary:
        actual_start = traversals[0]["from_collection"]
        out(f"  [AUTO-FIX] primary_collection={primary} but first traversal from={actual_start}, using {actual_start}")
        primary = actual_start

    primary_var = primary[0].lower()
    variables[primary] = primary_var
    aql_parts.append(f"FOR {primary_var} IN {primary}")

    # When query is Company -> MarketData and a specific ticker is provided, require ticker filter
    filters = json_plan.get("filters", {})
    plan_bind_vars = json_plan.get("bind_vars") or {}
    has_ticker_bind = bool(plan_bind_vars.get("ticker"))
    if primary == "Company" and traversals and any(t.get("to_collection") == "MarketData" for t in traversals):
        if has_ticker_bind and not any(k.strip() == "Company.ticker" for k in filters.keys()):
            out("  [AUTO-FIX] No Company.ticker filter; adding FILTER c.ticker == @ticker (bind_vars.ticker set)")
            aql_parts.append(f"  FILTER {primary_var}.ticker == @ticker")

    for trav in traversals:
        from_coll = trav.get("from_collection") or ""
        to_coll = trav.get("to_collection") or ""
        edge_coll = (trav.get("edge_collection") or "").strip()
        if not edge_coll or not to_coll:
            out(f"  [SKIP] Traversal missing edge_collection or to_collection - skipping")
            continue

        from_var = variables.get(from_coll)
        if not from_var:
            out(f"  [WARN] Traversal from {from_coll} but {from_coll} not in query path")
            from_var = from_coll[0].lower()
            counter = 2
            while from_var in variables.values():
                from_var = f"{from_coll[0].lower()}{counter}"
                counter += 1
            variables[from_coll] = from_var
            aql_parts.append(f"  FOR {from_var} IN {from_coll}")

        if to_coll in variables:
            out(f"  [SKIP] Duplicate traversal to {to_coll} (already in query path)")
            continue

        to_var = to_coll[0].lower()
        counter = 2
        while to_var in variables.values():
            to_var = f"{to_coll[0].lower()}{counter}"
            counter += 1
        variables[to_coll] = to_var
        edge_var = f"edge_{to_var}"
        aql_parts.append(f"  FOR {edge_var} IN {edge_coll}")
        aql_parts.append(f"    FILTER {edge_var}._from == {from_var}._id")
        aql_parts.append(f"    FOR {to_var} IN {to_coll}")
        aql_parts.append(f"      FILTER {to_var}._id == {edge_var}._to")

    if filters:
        for field_path, condition in filters.items():
            if "." not in field_path:
                continue
            collection, field = field_path.split(".", 1)
            var = variables.get(collection)
            if not var:
                out(f"  [SKIP] Filter on {collection}.{field} but {collection} not in query path")
                continue
            operator = condition.get("operator", "==")
            value = condition.get("value")
            indent = "      " if len(traversals) > 0 else "  "
            field_access = f"{var}['{field}']" if ("-" in field or " " in field) else f"{var}.{field}"
            # List value: emit FILTER field IN ["A", "B"] (e.g. multi-ticker compare)
            if isinstance(value, list) and len(value) > 0:
                aql_list = ", ".join(f'"{v}"' if isinstance(v, str) else str(v) for v in value)
                aql_parts.append(f"{indent}FILTER {field_access} IN [{aql_list}]")
                continue
            # Bind variable: value starting with @ is emitted as-is (e.g. @ticker)
            if isinstance(value, str) and value.startswith("@"):
                value = value
            elif isinstance(value, str):
                value = f'"{value}"'
            elif isinstance(value, bool):
                value = str(value).lower()
            elif isinstance(value, (int, float)):
                value = str(value)
            if operator == "CONTAINS":
                aql_parts.append(f"{indent}FILTER CONTAINS({field_access}, {value})")
            else:
                if "-" in field or " " in field:
                    aql_parts.append(f"{indent}FILTER {var}['{field}'] {operator} {value}")
                else:
                    aql_parts.append(f"{indent}FILTER {var}.{field} {operator} {value}")

    _agg = json_plan.get("aggregations") or json_plan.get("aggregation")
    aggregation = _agg[0] if isinstance(_agg, list) and _agg else (_agg if isinstance(_agg, dict) else None)
    _sort = json_plan.get("sort")
    sort_config = _sort[0] if isinstance(_sort, list) and _sort else (_sort if isinstance(_sort, dict) else None)

    if sort_config and not aggregation:
        field_path = sort_config.get("field", "")
        if "." in field_path:
            collection, field = field_path.split(".", 1)
            var = variables.get(collection)
            if not var:
                out(f"  [SKIP] Sort on {collection}.{field} but {collection} not in query path")
            else:
                direction = sort_config.get("direction", "DESC")
                indent = "      " if len(traversals) > 0 else "  "
                aql_parts.append(f"{indent}SORT {var}.{field} {direction}")

    if aggregation:
        agg_type = aggregation.get("type", "COUNT")
        agg_field = aggregation.get("field")
        group_by = aggregation.get("group_by", [])
        indent = "      " if len(traversals) > 0 else "  "

        if group_by:
            group_fields = []
            for field_path in group_by:
                if "." in field_path:
                    collection, field = field_path.split(".", 1)
                    var = variables.get(collection)
                    if not var:
                        out(f"  [SKIP] Group by {collection}.{field} but {collection} not in query path")
                        continue
                    group_fields.append(f"{field} = {var}.{field}")
            if group_fields:
                aql_parts.append(f"{indent}COLLECT {', '.join(group_fields)}")
            agg_added = False
            if agg_field and "." in agg_field:
                collection, field = agg_field.split(".", 1)
                var = variables.get(collection)
                if not var:
                    out(f"  [SKIP] Aggregate on {collection}.{field} but {collection} not in query path")
                else:
                    if agg_type == "COUNT":
                        aql_parts.append(f"{indent}AGGREGATE agg_count = LENGTH(1)")
                        agg_added = True
                    elif agg_type == "SUM":
                        aql_parts.append(f"{indent}AGGREGATE agg_sum = SUM({var}.{field})")
                        agg_added = True
                    elif agg_type == "AVG":
                        aql_parts.append(f"{indent}AGGREGATE agg_avg = AVG({var}.{field})")
                        agg_added = True
            if not agg_added and group_by:
                # COLLECT with group_by but no valid AGGREGATE: add count so RETURN has defined variable
                aql_parts.append(f"{indent}AGGREGATE agg_count = LENGTH(1)")
                agg_type = "COUNT"
        else:
            if agg_type == "COUNT":
                aql_parts.append(f"{indent}COLLECT WITH COUNT INTO total")

    if not aggregation:
        limit = json_plan.get("limit", 10)
        try:
            limit = int(limit) if limit is not None else 100
        except (TypeError, ValueError):
            limit = 100
        if limit <= 0 or limit > 1000:
            limit = min(max(1, limit), 1000)
            out(f"  [AUTO-FIX] Limit capped to {limit}")
        indent = "      " if len(traversals) > 0 else "  "
        aql_parts.append(f"{indent}LIMIT {limit}")

    return_fields = json_plan.get("return_fields", [])
    if isinstance(return_fields, dict):
        return_fields = list(return_fields) if return_fields else []
    if aggregation:
        indent = "      " if len(traversals) > 0 else "  "
        if aggregation.get("group_by"):
            return_obj = []
            for field_path in aggregation.get("group_by", []):
                if "." in field_path:
                    _, field = field_path.split(".", 1)
                    return_obj.append(f"{field}: {field}")
            agg_type = aggregation.get("type", "COUNT")
            aql_so_far = "\n".join(aql_parts)
            if agg_type == "COUNT" or "agg_count =" in aql_so_far:
                return_obj.append("count: agg_count")
            elif agg_type == "SUM" and "agg_sum =" in aql_so_far:
                return_obj.append("total: agg_sum")
            elif agg_type == "AVG" and "agg_avg =" in aql_so_far:
                return_obj.append("average: agg_avg")
            else:
                return_obj.append("count: agg_count")
            aql_parts.append(f"{indent}RETURN {{")
            for i, field_def in enumerate(return_obj):
                comma = "," if i < len(return_obj) - 1 else ""
                aql_parts.append(f"{indent}  {field_def}{comma}")
            aql_parts.append(f"{indent}}}")
        else:
            if any("COLLECT" in p and "total" in p for p in aql_parts):
                aql_parts.append(f"{indent}RETURN {{ count: total }}")
            else:
                indent = "      " if len(traversals) > 0 else "  "
                aql_parts.append(f"{indent}COLLECT WITH COUNT INTO total")
                aql_parts.append(f"{indent}RETURN {{ count: total }}")
    elif return_fields:
        return_obj = []
        for field_path in return_fields:
            if not isinstance(field_path, str) or "." not in field_path:
                continue
            collection, field = field_path.split(".", 1)
            var = variables.get(collection)
            if not var:
                out(f"  [SKIP] Return field {collection}.{field} but {collection} not in query path")
                continue
            return_obj.append(f"{field}: {var}.{field}")
        if return_obj:
            indent = "      " if len(traversals) > 0 else "  "
            aql_parts.append(f"{indent}RETURN {{")
            for i, field_def in enumerate(return_obj):
                comma = "," if i < len(return_obj) - 1 else ""
                aql_parts.append(f"{indent}  {field_def}{comma}")
            aql_parts.append(f"{indent}}}")
        else:
            indent = "      " if len(traversals) > 0 else "  "
            aql_parts.append(f"{indent}RETURN {primary_var}")
    else:
        indent = "      " if len(traversals) > 0 else "  "
        aql_parts.append(f"{indent}RETURN {primary_var}")

    return "\n".join(aql_parts)
