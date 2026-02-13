"""
UK GPDO Legislation Parser
==========================
Parses the Town and Country Planning (General Permitted Development) (England)
Order 2015 (SI 2015/596) from legislation.gov.uk CLML XML format into flat CSV
tables suitable for graph/knowledge-graph ingestion (e.g. SynaLinks).

Usage:
    python parse_gpdo.py

This will:
1. Download the GPDO XML from legislation.gov.uk
2. Parse it into 4 flat CSV tables:
   - articles.csv        (all provisions with full hierarchy)
   - cross_references.csv (all citations/references between provisions)
   - definitions.csv      (defined terms)
   - hierarchy.csv        (parent-child relationships between provisions)
3. Save them to an output directory

Requirements:
    pip install requests lxml pandas
"""

import requests
import pandas as pd
from lxml import etree
from pathlib import Path
import re
import hashlib
import time
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# The GPDO in its latest revised form (CLML XML)
GPDO_URL = "https://www.legislation.gov.uk/uksi/2015/596/data.xml"

# If you want a specific point-in-time version, use e.g.:
# GPDO_URL = "https://www.legislation.gov.uk/uksi/2015/596/2025-05-29/data.xml"

OUTPUT_DIR = Path("output")

# CLML namespaces used in legislation.gov.uk XML
NAMESPACES = {
    "leg": "http://www.legislation.gov.uk/namespaces/legislation",
    "ukm": "http://www.legislation.gov.uk/namespaces/metadata",
    "dc": "http://purl.org/dc/elements/1.1/",
    "atom": "http://www.w3.org/2005/Atom",
    "xhtml": "http://www.w3.org/1999/xhtml",
    "math": "http://www.w3.org/1998/Math/MathML",
}

# Legislation type for this instrument
LEGISLATION_ID = "uksi/2015/596"
LEGISLATION_TITLE = "The Town and Country Planning (General Permitted Development) (England) Order 2015"
LEGISLATION_SHORT = "GPDO 2015"

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_xml(url: str, cache_path: Path = Path("gpdo_raw.xml")) -> bytes:
    """Download the CLML XML, with simple file caching."""
    if cache_path.exists():
        print(f"Using cached XML from {cache_path}")
        return cache_path.read_bytes()

    print(f"Downloading from {url} ...")
    headers = {
        "User-Agent": "LegislationParser/1.0 (research; contact@example.com)",
        "Accept": "application/xml",
    }
    resp = requests.get(url, headers=headers, timeout=120)

    # legislation.gov.uk may return 202 Accepted for large docs (needs retry)
    retries = 0
    while resp.status_code == 202 and retries < 5:
        print(f"  Server returned 202 (generating), retrying in 15s... (attempt {retries+1})")
        time.sleep(15)
        resp = requests.get(url, headers=headers, timeout=120)
        retries += 1

    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    print(f"  Saved {len(resp.content)} bytes to {cache_path}")
    return resp.content


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def text_of(element) -> str:
    """Extract all text content from an element, stripping whitespace."""
    if element is None:
        return ""
    parts = []
    for t in element.itertext():
        parts.append(t.strip())
    return " ".join(p for p in parts if p)


def make_id(*parts) -> str:
    """Create a stable provision ID from hierarchy parts."""
    return "/".join(str(p) for p in parts if p)


def short_hash(text: str) -> str:
    """Short hash for deduplication."""
    return hashlib.md5(text.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_provisions(root) -> list[dict]:
    """
    Walk the CLML tree and extract every provision (article, paragraph,
    sub-paragraph) with its full hierarchy context.

    CLML hierarchy for Statutory Instruments:
        <Body>
            <Part>             -> Order articles (1-12)
                <Pblock>       -> Grouped articles
                    <P1>       -> Article / numbered provision
                        <P2>   -> Sub-paragraph
                            <P3> -> Sub-sub-paragraph
        <Schedules>
            <Schedule>         -> Schedule 1, 2, etc.
                <Part>         -> Part 1, 2, ...
                    <Pblock>   -> Class A, B, C, ... (for GPDO)
                        <P1>   -> Numbered paragraph (A.1, A.2, ...)
                            <P2>  -> Sub-paragraph
                                <P3> -> Sub-sub-paragraph
    """
    provisions = []

    def walk(element, context: dict):
        """Recursively walk the XML tree, building up hierarchy context."""
        tag = etree.QName(element.tag).localname if isinstance(element.tag, str) else ""

        # Copy context so siblings don't pollute each other
        ctx = dict(context)

        # --- Structural elements that define hierarchy ---

        if tag == "Body":
            ctx["section_type"] = "body"
            for child in element:
                walk(child, ctx)
            return

        if tag == "Schedules":
            ctx["section_type"] = "schedules"
            for child in element:
                walk(child, ctx)
            return

        if tag == "Schedule":
            num_el = element.find("leg:Number", NAMESPACES)
            title_el = element.find("leg:TitleBlock/leg:Title", NAMESPACES)
            if title_el is None:
                title_el = element.find("leg:TitleBlock", NAMESPACES)
            ctx["schedule_number"] = text_of(num_el)
            ctx["schedule_title"] = text_of(title_el)
            ctx["schedule_id"] = element.get("id", "")
            ctx["schedule_uri"] = element.get(
                "{http://www.legislation.gov.uk/namespaces/legislation}IdURI",
                element.get("IdURI", "")
            )
            for child in element:
                walk(child, ctx)
            return

        if tag == "Part":
            num_el = element.find("leg:Number", NAMESPACES)
            title_el = element.find("leg:Title", NAMESPACES)
            ctx["part_number"] = text_of(num_el)
            ctx["part_title"] = text_of(title_el)
            ctx["part_id"] = element.get("id", "")
            ctx["part_uri"] = element.get("IdURI", "")
            for child in element:
                walk(child, ctx)
            return

        if tag == "Chapter":
            num_el = element.find("leg:Number", NAMESPACES)
            title_el = element.find("leg:Title", NAMESPACES)
            ctx["chapter_number"] = text_of(num_el)
            ctx["chapter_title"] = text_of(title_el)
            for child in element:
                walk(child, ctx)
            return

        if tag == "Pblock":
            title_el = element.find("leg:Title", NAMESPACES)
            ctx["class_block"] = text_of(title_el)
            ctx["class_block_id"] = element.get("id", "")
            for child in element:
                walk(child, ctx)
            return

        if tag == "PsubBlock":
            title_el = element.find("leg:Title", NAMESPACES)
            ctx["sub_block"] = text_of(title_el)
            ctx["sub_block_id"] = element.get("id", "")
            for child in element:
                walk(child, ctx)
            return

        # --- Leaf provisions (the actual articles / paragraphs) ---

        if tag == "P1":
            pn = element.find("leg:Pnumber", NAMESPACES)
            p1_number = text_of(pn)
            p1_id = element.get("id", "")
            p1_uri = element.get("IdURI", "")

            # Get the direct text of P1 (in P1para)
            p1_text_parts = []
            p2_elements = []
            for child in element:
                child_tag = etree.QName(child.tag).localname if isinstance(child.tag, str) else ""
                if child_tag == "P1para":
                    # P1para may contain direct text and/or P2 children
                    for sub in child:
                        sub_tag = etree.QName(sub.tag).localname if isinstance(sub.tag, str) else ""
                        if sub_tag == "P2":
                            p2_elements.append(sub)
                        elif sub_tag == "Text":
                            p1_text_parts.append(text_of(sub))
                        else:
                            p1_text_parts.append(text_of(sub))
                elif child_tag == "P2":
                    p2_elements.append(child)

            p1_text = " ".join(p1_text_parts).strip()

            # Record this P1 provision
            provision = {
                "provision_id": p1_id,
                "provision_uri": p1_uri,
                "legislation_id": LEGISLATION_ID,
                "legislation_title": LEGISLATION_SHORT,
                "section_type": ctx.get("section_type", ""),
                "schedule_number": ctx.get("schedule_number", ""),
                "schedule_title": ctx.get("schedule_title", ""),
                "part_number": ctx.get("part_number", ""),
                "part_title": ctx.get("part_title", ""),
                "chapter_number": ctx.get("chapter_number", ""),
                "chapter_title": ctx.get("chapter_title", ""),
                "class_block": ctx.get("class_block", ""),
                "sub_block": ctx.get("sub_block", ""),
                "provision_number": p1_number,
                "provision_level": "P1",
                "parent_provision_id": ctx.get("parent_p1_id", ""),
                "text": p1_text,
                "status": element.get("Status", "valid"),
                "restrict_start_date": element.get("RestrictStartDate", ""),
                "restrict_end_date": element.get("RestrictEndDate", ""),
                "restrict_extent": element.get("RestrictExtent", ""),
            }
            provisions.append(provision)

            # Now recurse into P2 children
            p2_ctx = dict(ctx)
            p2_ctx["parent_p1_id"] = p1_id
            for p2 in p2_elements:
                _parse_p2(p2, p2_ctx, provisions)

            return

        # For any other element, just recurse
        for child in element:
            walk(child, ctx)

    def _parse_p2(element, ctx, provisions):
        """Parse a P2 (sub-paragraph) element."""
        pn = element.find("leg:Pnumber", NAMESPACES)
        p2_number = text_of(pn)
        p2_id = element.get("id", "")
        p2_uri = element.get("IdURI", "")

        # Gather text and P3 children
        p2_text_parts = []
        p3_elements = []
        for child in element:
            child_tag = etree.QName(child.tag).localname if isinstance(child.tag, str) else ""
            if child_tag == "P2para":
                for sub in child:
                    sub_tag = etree.QName(sub.tag).localname if isinstance(sub.tag, str) else ""
                    if sub_tag == "P3":
                        p3_elements.append(sub)
                    elif sub_tag == "Text":
                        p2_text_parts.append(text_of(sub))
                    else:
                        p2_text_parts.append(text_of(sub))
            elif child_tag == "P3":
                p3_elements.append(child)

        p2_text = " ".join(p2_text_parts).strip()

        provision = {
            "provision_id": p2_id,
            "provision_uri": p2_uri,
            "legislation_id": LEGISLATION_ID,
            "legislation_title": LEGISLATION_SHORT,
            "section_type": ctx.get("section_type", ""),
            "schedule_number": ctx.get("schedule_number", ""),
            "schedule_title": ctx.get("schedule_title", ""),
            "part_number": ctx.get("part_number", ""),
            "part_title": ctx.get("part_title", ""),
            "chapter_number": ctx.get("chapter_number", ""),
            "chapter_title": ctx.get("chapter_title", ""),
            "class_block": ctx.get("class_block", ""),
            "sub_block": ctx.get("sub_block", ""),
            "provision_number": p2_number,
            "provision_level": "P2",
            "parent_provision_id": ctx.get("parent_p1_id", ""),
            "text": p2_text,
            "status": element.get("Status", "valid"),
            "restrict_start_date": element.get("RestrictStartDate", ""),
            "restrict_end_date": element.get("RestrictEndDate", ""),
            "restrict_extent": element.get("RestrictExtent", ""),
        }
        provisions.append(provision)

        # P3 children
        p3_ctx = dict(ctx)
        p3_ctx["parent_p1_id"] = p2_id  # re-use field for parent chain
        for p3 in p3_elements:
            _parse_p3(p3, p3_ctx, provisions)

    def _parse_p3(element, ctx, provisions):
        """Parse a P3 (sub-sub-paragraph) element."""
        pn = element.find("leg:Pnumber", NAMESPACES)
        p3_number = text_of(pn)
        p3_id = element.get("id", "")
        p3_uri = element.get("IdURI", "")

        p3_text_parts = []
        p4_elements = []
        for child in element:
            child_tag = etree.QName(child.tag).localname if isinstance(child.tag, str) else ""
            if child_tag == "P3para":
                for sub in child:
                    sub_tag = etree.QName(sub.tag).localname if isinstance(sub.tag, str) else ""
                    if sub_tag == "P4":
                        p4_elements.append(sub)
                    elif sub_tag == "Text":
                        p3_text_parts.append(text_of(sub))
                    else:
                        p3_text_parts.append(text_of(sub))
            elif child_tag == "P4":
                p4_elements.append(child)

        p3_text = " ".join(p3_text_parts).strip()

        provision = {
            "provision_id": p3_id,
            "provision_uri": p3_uri,
            "legislation_id": LEGISLATION_ID,
            "legislation_title": LEGISLATION_SHORT,
            "section_type": ctx.get("section_type", ""),
            "schedule_number": ctx.get("schedule_number", ""),
            "schedule_title": ctx.get("schedule_title", ""),
            "part_number": ctx.get("part_number", ""),
            "part_title": ctx.get("part_title", ""),
            "chapter_number": ctx.get("chapter_number", ""),
            "chapter_title": ctx.get("chapter_title", ""),
            "class_block": ctx.get("class_block", ""),
            "sub_block": ctx.get("sub_block", ""),
            "provision_number": p3_number,
            "provision_level": "P3",
            "parent_provision_id": ctx.get("parent_p1_id", ""),
            "text": p3_text,
            "status": element.get("Status", "valid"),
            "restrict_start_date": element.get("RestrictStartDate", ""),
            "restrict_end_date": element.get("RestrictEndDate", ""),
            "restrict_extent": element.get("RestrictExtent", ""),
        }
        provisions.append(provision)

        # P4 if they exist (rare but possible)
        for p4 in p4_elements:
            _parse_p4(p4, {**ctx, "parent_p1_id": p3_id}, provisions)

    def _parse_p4(element, ctx, provisions):
        """Parse a P4 element (rare, deep nesting)."""
        pn = element.find("leg:Pnumber", NAMESPACES)
        p4_number = text_of(pn)
        p4_id = element.get("id", "")
        p4_uri = element.get("IdURI", "")

        p4_text = text_of(element)

        provision = {
            "provision_id": p4_id,
            "provision_uri": p4_uri,
            "legislation_id": LEGISLATION_ID,
            "legislation_title": LEGISLATION_SHORT,
            "section_type": ctx.get("section_type", ""),
            "schedule_number": ctx.get("schedule_number", ""),
            "schedule_title": ctx.get("schedule_title", ""),
            "part_number": ctx.get("part_number", ""),
            "part_title": ctx.get("part_title", ""),
            "chapter_number": ctx.get("chapter_number", ""),
            "chapter_title": ctx.get("chapter_title", ""),
            "class_block": ctx.get("class_block", ""),
            "sub_block": ctx.get("sub_block", ""),
            "provision_number": p4_number,
            "provision_level": "P4",
            "parent_provision_id": ctx.get("parent_p1_id", ""),
            "text": p4_text,
            "status": element.get("Status", "valid"),
            "restrict_start_date": element.get("RestrictStartDate", ""),
            "restrict_end_date": element.get("RestrictEndDate", ""),
            "restrict_extent": element.get("RestrictExtent", ""),
        }
        provisions.append(provision)

    # Start the walk from the root
    # Find the Legislation element (may be nested under wrapper)
    leg_el = root.find(".//leg:Body", NAMESPACES)
    if leg_el is not None:
        walk(leg_el, {})

    sched_el = root.find(".//leg:Schedules", NAMESPACES)
    if sched_el is not None:
        walk(sched_el, {})

    return provisions


def parse_cross_references(root) -> list[dict]:
    """
    Extract all <Citation> elements from the XML.
    These are inline references from one provision to another piece of legislation
    or another section within the same instrument.
    """
    refs = []

    for citation in root.iter("{http://www.legislation.gov.uk/namespaces/legislation}Citation"):
        # Find the nearest ancestor with an id to determine source provision
        source_id = ""
        parent = citation.getparent()
        while parent is not None:
            pid = parent.get("id", "")
            if pid:
                source_id = pid
                break
            parent = parent.getparent()

        target_uri = citation.get("URI", "")
        target_id = citation.get("id", "")
        citation_text = text_of(citation)

        # Determine if internal or external reference
        is_internal = LEGISLATION_ID in target_uri if target_uri else False

        # Extract target legislation info from URI
        # URIs look like: http://www.legislation.gov.uk/id/ukpga/1990/8/section/59
        target_legislation = ""
        target_provision = ""
        if target_uri:
            parts = target_uri.replace("http://www.legislation.gov.uk/id/", "").split("/")
            if len(parts) >= 3:
                target_legislation = "/".join(parts[:3])  # e.g. ukpga/1990/8
                if len(parts) > 3:
                    target_provision = "/".join(parts[3:])  # e.g. section/59

        ref = {
            "source_provision_id": source_id,
            "target_uri": target_uri,
            "target_legislation_id": target_legislation,
            "target_provision_path": target_provision,
            "citation_text": citation_text,
            "is_internal": is_internal,
            "reference_type": "citation",
        }
        refs.append(ref)

    # Also extract CommentaryRef and FootnoteRef for amendment tracking
    for cref in root.iter("{http://www.legislation.gov.uk/namespaces/legislation}CommentaryRef"):
        source_id = ""
        parent = cref.getparent()
        while parent is not None:
            pid = parent.get("id", "")
            if pid:
                source_id = pid
                break
            parent = parent.getparent()

        ref_id = cref.get("Ref", "")
        refs.append({
            "source_provision_id": source_id,
            "target_uri": "",
            "target_legislation_id": "",
            "target_provision_path": "",
            "citation_text": f"Commentary: {ref_id}",
            "is_internal": True,
            "reference_type": "commentary",
        })

    return refs


def parse_definitions(root) -> list[dict]:
    """
    Extract defined terms from the legislation.
    In CLML, definitions often appear in <P1> elements within interpretation
    sections, marked with <Term> elements.
    """
    definitions = []

    for term_el in root.iter("{http://www.legislation.gov.uk/namespaces/legislation}Term"):
        term_text = text_of(term_el)
        term_id = term_el.get("id", "")

        # Find the containing provision
        source_id = ""
        parent = term_el.getparent()
        while parent is not None:
            pid = parent.get("id", "")
            if pid:
                source_id = pid
                break
            parent = parent.getparent()

        # Try to get the definition text (usually the parent Text element or P1para)
        def_text = ""
        text_parent = term_el.getparent()
        if text_parent is not None:
            def_text = text_of(text_parent)

        definitions.append({
            "term": term_text,
            "term_id": term_id,
            "definition_text": def_text,
            "source_provision_id": source_id,
            "legislation_id": LEGISLATION_ID,
        })

    return definitions


def build_hierarchy(provisions: list[dict]) -> list[dict]:
    """
    Build explicit parent-child relationships from the provisions table.
    This creates edges for a graph between:
    - legislation -> schedule
    - schedule -> part
    - part -> class_block
    - class_block -> P1
    - P1 -> P2
    - P2 -> P3
    etc.
    """
    hierarchy = []
    seen = set()

    for prov in provisions:
        # Legislation -> Schedule
        if prov["schedule_number"]:
            key = (LEGISLATION_ID, "has_schedule", prov["schedule_number"])
            if key not in seen:
                seen.add(key)
                hierarchy.append({
                    "parent_type": "legislation",
                    "parent_id": LEGISLATION_ID,
                    "parent_label": LEGISLATION_SHORT,
                    "relationship": "has_schedule",
                    "child_type": "schedule",
                    "child_id": prov["schedule_number"],
                    "child_label": f"{prov['schedule_number']} - {prov['schedule_title']}",
                })

        # Schedule -> Part
        if prov["schedule_number"] and prov["part_number"]:
            key = (prov["schedule_number"], "has_part", prov["part_number"])
            if key not in seen:
                seen.add(key)
                hierarchy.append({
                    "parent_type": "schedule",
                    "parent_id": prov["schedule_number"],
                    "parent_label": prov["schedule_number"],
                    "relationship": "has_part",
                    "child_type": "part",
                    "child_id": prov["part_number"],
                    "child_label": f"{prov['part_number']} - {prov['part_title']}",
                })

        # Part -> Class block
        if prov["part_number"] and prov["class_block"]:
            key = (prov["part_number"], "has_class", prov["class_block"])
            if key not in seen:
                seen.add(key)
                hierarchy.append({
                    "parent_type": "part",
                    "parent_id": prov["part_number"],
                    "parent_label": prov["part_number"],
                    "relationship": "has_class",
                    "child_type": "class_block",
                    "child_id": prov["class_block"],
                    "child_label": prov["class_block"],
                })

        # Class block (or part) -> Provision
        if prov["provision_level"] == "P1":
            parent_key = prov["class_block"] or prov["part_number"] or prov["schedule_number"]
            parent_type = "class_block" if prov["class_block"] else "part" if prov["part_number"] else "schedule"
            key = (parent_key, "has_provision", prov["provision_id"])
            if key not in seen and prov["provision_id"]:
                seen.add(key)
                hierarchy.append({
                    "parent_type": parent_type,
                    "parent_id": parent_key,
                    "parent_label": parent_key,
                    "relationship": "has_provision",
                    "child_type": "provision",
                    "child_id": prov["provision_id"],
                    "child_label": f"{prov['provision_number']}",
                })

        # P1 -> P2, P2 -> P3, etc.
        if prov["parent_provision_id"] and prov["provision_level"] in ("P2", "P3", "P4"):
            key = (prov["parent_provision_id"], "has_sub_provision", prov["provision_id"])
            if key not in seen and prov["provision_id"]:
                seen.add(key)
                hierarchy.append({
                    "parent_type": "provision",
                    "parent_id": prov["parent_provision_id"],
                    "parent_label": prov["parent_provision_id"],
                    "relationship": "has_sub_provision",
                    "child_type": "provision",
                    "child_id": prov["provision_id"],
                    "child_label": f"{prov['provision_number']}",
                })

    return hierarchy


def parse_metadata(root) -> dict:
    """Extract top-level metadata about the legislation."""
    meta = {}
    metadata_el = root.find(".//ukm:Metadata", NAMESPACES)
    if metadata_el is not None:
        title_el = metadata_el.find(".//dc:title", NAMESPACES)
        if title_el is not None:
            meta["title"] = text_of(title_el)

        # Document type, year, number
        doc_main = metadata_el.find(".//ukm:DocumentMainType", NAMESPACES)
        if doc_main is not None:
            meta["document_type"] = doc_main.get("Value", "")

        year_el = metadata_el.find(".//ukm:Year", NAMESPACES)
        if year_el is not None:
            meta["year"] = year_el.get("Value", "")

        number_el = metadata_el.find(".//ukm:Number", NAMESPACES)
        if number_el is not None:
            meta["number"] = number_el.get("Value", "")

    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Download
    xml_bytes = download_xml(GPDO_URL)

    # 2. Parse XML
    print("Parsing XML...")
    root = etree.fromstring(xml_bytes)

    # 3. Extract metadata
    meta = parse_metadata(root)
    print(f"  Document: {meta.get('title', 'unknown')}")
    print(f"  Type: {meta.get('document_type', '?')}, Year: {meta.get('year', '?')}, Number: {meta.get('number', '?')}")

    # 4. Extract provisions
    print("Extracting provisions...")
    provisions = parse_provisions(root)
    print(f"  Found {len(provisions)} provisions")

    # 5. Extract cross-references
    print("Extracting cross-references...")
    cross_refs = parse_cross_references(root)
    print(f"  Found {len(cross_refs)} cross-references")

    # 6. Extract definitions
    print("Extracting definitions...")
    definitions = parse_definitions(root)
    print(f"  Found {len(definitions)} defined terms")

    # 7. Build hierarchy
    print("Building hierarchy...")
    hierarchy = build_hierarchy(provisions)
    print(f"  Found {len(hierarchy)} hierarchy edges")

    # 8. Save to CSV
    print("Saving CSVs...")

    df_provisions = pd.DataFrame(provisions)
    df_provisions.to_csv(OUTPUT_DIR / "articles.csv", index=False)
    print(f"  articles.csv: {len(df_provisions)} rows, {len(df_provisions.columns)} columns")

    df_refs = pd.DataFrame(cross_refs)
    df_refs.to_csv(OUTPUT_DIR / "cross_references.csv", index=False)
    print(f"  cross_references.csv: {len(df_refs)} rows")

    df_defs = pd.DataFrame(definitions)
    df_defs.to_csv(OUTPUT_DIR / "definitions.csv", index=False)
    print(f"  definitions.csv: {len(df_defs)} rows")

    df_hier = pd.DataFrame(hierarchy)
    df_hier.to_csv(OUTPUT_DIR / "hierarchy.csv", index=False)
    print(f"  hierarchy.csv: {len(df_hier)} rows")

    # 9. Print summary / sample
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUT")
    print("=" * 60)

    if len(df_provisions) > 0:
        print("\n--- articles.csv columns ---")
        print(list(df_provisions.columns))
        print("\n--- First 5 rows (key columns) ---")
        key_cols = [
            "provision_id", "section_type", "schedule_number", "part_number",
            "class_block", "provision_number", "provision_level", "text"
        ]
        existing_cols = [c for c in key_cols if c in df_provisions.columns]
        print(df_provisions[existing_cols].head().to_string(max_colwidth=60))

    if len(df_refs) > 0:
        print("\n--- cross_references.csv (first 5) ---")
        print(df_refs.head().to_string(max_colwidth=50))

    if len(df_defs) > 0:
        print("\n--- definitions.csv (first 5) ---")
        print(df_defs.head().to_string(max_colwidth=60))

    print(f"\nAll files saved to: {OUTPUT_DIR.resolve()}")
    print("Done!")


if __name__ == "__main__":
    main()