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
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GPDO_URL = "https://www.legislation.gov.uk/uksi/2015/596/data.xml"
TCPA_ID = "ukpga/1990/8"
TCPA_URL = "https://www.legislation.gov.uk/ukpga/1990/8/data.xml"

OUTPUT_DIR = Path("output")
CACHE_DIR = OUTPUT_DIR / "cache"
METADATA_FETCH_TIMEOUT_SECONDS = 15
METADATA_FETCH_WORKERS = 8

NAMESPACES = {
    "leg": "http://www.legislation.gov.uk/namespaces/legislation",
    "ukm": "http://www.legislation.gov.uk/namespaces/metadata",
    "dc": "http://purl.org/dc/elements/1.1/",
    "atom": "http://www.w3.org/2005/Atom",
    "xhtml": "http://www.w3.org/1999/xhtml",
    "math": "http://www.w3.org/1998/Math/MathML",
}

LEGISLATION_ID = "uksi/2015/596"
LEGISLATION_TITLE = (
    "The Town and Country Planning (General Permitted Development) (England) Order 2015"
)
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

    retries = 0
    while resp.status_code == 202 and retries < 5:
        print(
            f"  Server returned 202 (generating), retrying in 15s... (attempt {retries+1})"
        )
        time.sleep(15)
        resp = requests.get(url, headers=headers, timeout=120)
        retries += 1

    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    print(f"  Saved {len(resp.content)} bytes to {cache_path}")
    return resp.content


def download_xml_metadata(url: str, cache_path: Path, verbose: bool = False) -> dict:
    """
    Download CLML XML and extract lightweight metadata (title/type/year/number).
    Returns {} on failure.
    """
    if cache_path.exists():
        try:
            root = etree.fromstring(cache_path.read_bytes())
            return parse_metadata(root)
        except Exception:
            pass

    headers = {
        "User-Agent": "LegislationParser/1.0 (research; contact@example.com)",
        "Accept": "application/xml",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=METADATA_FETCH_TIMEOUT_SECONDS)
        retries = 0
        while resp.status_code == 202 and retries < 5:
            time.sleep(10)
            resp = requests.get(
                url, headers=headers, timeout=METADATA_FETCH_TIMEOUT_SECONDS
            )
            retries += 1
        resp.raise_for_status()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(resp.content)
        root = etree.fromstring(resp.content)
        return parse_metadata(root)
    except Exception as e:
        if verbose:
            print(f"  Warning: failed to fetch XML {url}: {e}")
        return {}


def resolve_canonical_legislation_uri(legislation_id: str) -> str:
    """
    Resolve /id/{id}/ to the canonical legislation document URI.
    This handles historical/regnal redirects (e.g. /ukpga/Geo5and1Edw8/26/49/contents).
    """
    cache_base = legislation_id.replace("/", "_")
    cache_path = CACHE_DIR / f"{cache_base}_canonical_uri.txt"
    if cache_path.exists():
        cached = cache_path.read_text(encoding="utf-8").strip()
        if cached:
            return cached

    id_uri = f"https://www.legislation.gov.uk/id/{legislation_id}/"
    headers = {
        "User-Agent": "LegislationParser/1.0 (research; contact@example.com)",
        "Accept": "text/html,application/xhtml+xml",
    }
    try:
        resp = requests.get(
            id_uri,
            headers=headers,
            timeout=METADATA_FETCH_TIMEOUT_SECONDS,
            allow_redirects=True,
        )
        resp.raise_for_status()
        final_url = resp.url.rstrip("/")
    except Exception:
        final_url = f"https://www.legislation.gov.uk/{legislation_id}"

    # Drop /id prefix if still present.
    final_url = final_url.replace("https://www.legislation.gov.uk/id/", "https://www.legislation.gov.uk/")

    # Normalise away content pages to document root.
    final_url = re.sub(r"/contents(?:/enacted)?$", "", final_url)
    final_url = re.sub(r"/enacted$", "", final_url)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(final_url, encoding="utf-8")
    return final_url


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def text_of(element) -> str:
    """Extract all text content from an element, stripping excess whitespace."""
    if element is None:
        return ""
    parts = []
    for t in element.itertext():
        stripped = t.strip()
        if stripped:
            parts.append(stripped)
    return " ".join(parts)


def pnumber_text(pn_el, element) -> str:
    """
    Return the canonical display number for a provision.

    CLML stores paragraph numbers in several ways:
      - <Pnumber>A</Pnumber>                                     -> "A"
      - <Pnumber PuncAfter=".1">A</Pnumber>                     -> "A.1"
      - <Pnumber PuncAfter="A.">B</Pnumber>                     -> "BA"   (Class BA)
      - <Pnumber PuncAfter="A.1">B</Pnumber>                    -> "BA.1" (Class BA, para 1)
      - <Pnumber PuncAfter="."><Addition>AA.1</Addition></Pnumber> -> "AA.1"
      - <Pnumber PuncAfter="."><Addition>h</Addition></Pnumber>    -> "h"

    Rule: itertext() gives the base text (handles <Addition> children);
    append PuncAfter in full, then strip any trailing ".".

    We deliberately do NOT use shortId here: shortIds are sometimes stale
    after amendments (e.g. a provision added as "(h)" by amendment may still
    carry shortId ending in "-c" from the original numbering).
    """
    if pn_el is None:
        return ""
    text = "".join(pn_el.itertext()).strip()
    punc = pn_el.get("PuncAfter", "")
    if punc:
        result = text + punc
        # Strip trailing period (display punctuation, not part of the number)
        return result.rstrip(".")
    return text


def normalise_schedule(schedule_number: str) -> str:
    """'SCHEDULE 2' -> 'sch2'"""
    m = re.search(r"\d+\w*", schedule_number)
    return f"sch{m.group().lower()}" if m else "sch"


def normalise_part(part_number: str) -> str:
    """'PART 12A' -> 'pt12a'"""
    m = re.search(r"\d+\w*", part_number)
    return f"pt{m.group().lower()}" if m else "pt"


def build_provision_id(
    ctx: dict, provision_number: str, parent_id: str, xml_id: str = ""
) -> str:
    """
    Build a short, globally-unique provision ID from hierarchy context.

    shortIds and full XML ids in the CLML are NOT globally unique — the same
    shortId (e.g. 'schedule-2-paragraph-A-a') is reused across all 20 Parts of
    Schedule 2. We construct our own IDs that embed schedule/part/class context.

    Body articles:    art-3 / art-3-1 / art-3-1-a / art-7ZA
    Schedule P1:      sch2-pt1-A / sch2-pt1-A.1   (class prefix usually redundant as it matches para letter)
                      sch2-pt12-clsB-B / sch2-pt12-clsBA-B  (class prefix added when needed for uniqueness)
    Schedule P2+:     {parent_id}-{provision_number}

    Edge cases:
    - Revoked provisions have empty Pnumber; fall back to last segment of XML id.
    - Classes with the same paragraph letter (e.g. Class B and Class BA both have para "B"):
      include class abbreviation as a prefix to the paragraph number.
    """
    section_type = ctx.get("section_type", "")
    num = provision_number.strip()

    # Fallback for empty numbers (revoked/omitted provisions)
    if not num:
        if xml_id:
            # Use the last meaningful segment of the full XML id
            segs = [s for s in xml_id.split("-") if s]
            num = segs[-1] if segs else "unknown"
        else:
            num = "unknown"

    if section_type == "body":
        if not parent_id:
            return f"art-{num}"
        else:
            return f"{parent_id}-{num}"

    # Schedule provisions — P1 only (parent_id is empty)
    if not parent_id:
        parts = []
        if ctx.get("schedule_number"):
            parts.append(normalise_schedule(ctx["schedule_number"]))
        if ctx.get("part_number"):
            parts.append(normalise_part(ctx["part_number"]))

        # Add class abbreviation prefix when present, to avoid collisions
        # between different Classes in the same Part sharing the same para letter
        # (e.g. Class B and Class BA in Part 12 both have paragraph "B").
        if ctx.get("class_block"):
            m = re.match(r"Class ([A-Z]+[A-Z0-9]*)", ctx["class_block"], re.IGNORECASE)
            if m:
                cls_code = m.group(1).upper()
                # Only include cls prefix if there's a risk of collision:
                # i.e. the class code does NOT start with the same letters as the para number.
                # In the normal case (Class A → para A, Class AA → para AA) the para number
                # already disambiguates. Only include when they could collide.
                # Simple heuristic: include cls prefix always for safety.
                parts.append(f"cls{cls_code}")

        parts.append(num)
        return "-".join(parts)
    else:
        # P2+ — extend parent ID with this number
        return f"{parent_id}-{num}"


def build_full_citation(ctx: dict) -> str:
    """
    Build a human-readable legal citation string as a lawyer or court would use.

    citation_parts in ctx holds the FULL path including the current node's number:
      Body P1 article 3:          citation_parts=["3"]         -> "article 3"
      Body P2 article 3(1):       citation_parts=["3","1"]     -> "article 3(1)"
      Body P3 article 3(1)(a):    citation_parts=["3","1","a"] -> "article 3(1)(a)"

      Schedule P1 paragraph A.1:  citation_parts=["A.1"]       -> "Schedule 2, Part 1, Class A, paragraph A.1"
      Schedule P3 A.1(a):         citation_parts=["A.1","a"]   -> "..., paragraph A.1(a)"
      Schedule P4 A.1(a)(i):      citation_parts=["A.1","a","i"] -> "..., paragraph A.1(a)(i)"
    """
    citation_parts = ctx.get("citation_parts", [])
    section_type = ctx.get("section_type", "")

    if not citation_parts:
        return ""

    if section_type == "body":
        article = citation_parts[0]
        subs = "".join(f"({p})" for p in citation_parts[1:])
        return f"article {article}{subs}"

    # Schedule provisions
    components = []
    if ctx.get("schedule_number"):
        components.append(ctx["schedule_number"].title())
    if ctx.get("part_number"):
        components.append(ctx["part_number"].title())
    if ctx.get("class_block"):
        m = re.match(r"(Class [A-Z]+[A-Z0-9]*)", ctx["class_block"], re.IGNORECASE)
        if m:
            components.append(m.group(1))

    prefix = ", ".join(components)

    para_ref = f"paragraph {citation_parts[0]}" + "".join(
        f"({p})" for p in citation_parts[1:]
    )

    if prefix:
        return f"{prefix}, {para_ref}"
    return para_ref


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------


def parse_provisions(root) -> tuple[list[dict], dict]:
    """
    Walk the CLML tree and extract every provision with its full hierarchy.

    Returns:
        (provisions, structural_ids) where structural_ids maps the XML element
        id of every structural container (Schedule, Part, Pblock) to its
        normalised node id. This is merged into the xmlid_to_provid lookup so
        that cross-references inside structural containers resolve correctly.
    """
    provisions = []
    structural_ids: dict = {}  # xml element id -> normalised structural node id

    LEG = "http://www.legislation.gov.uk/namespaces/legislation"

    # Tags that represent sub-provisions at each level
    SUB_TAGS = {"P2", "P3", "P4", "P5"}

    def get_tag(element) -> str:
        return (
            etree.QName(element.tag).localname if isinstance(element.tag, str) else ""
        )

    def direct_text(para_element) -> str:
        """
        Extract only the direct <Text> content from a P*para element,
        NOT recursing into any nested P* sub-provision children.
        """
        parts = []
        for child in para_element:
            tag = get_tag(child)
            if tag in SUB_TAGS:
                continue  # these become their own rows
            # For Text nodes and inline elements (Addition, Substitution, etc.)
            parts.append(text_of(child))
        return " ".join(p for p in parts if p).strip()

    def collect_sub_provisions(para_element) -> list:
        """Return all direct P2/P3/P4/P5 children of a P*para element."""
        children = []
        for child in para_element:
            if get_tag(child) in SUB_TAGS:
                children.append(child)
        return children

    def make_provision(element, ctx: dict, level: str, parent_id: str) -> dict:
        """Build a provision dict from an element and context."""
        pn_el = element.find(f"{{{LEG}}}Pnumber")
        provision_number = pnumber_text(pn_el, element)
        xml_id = element.get("id", "")
        provision_id = build_provision_id(ctx, provision_number, parent_id, xml_id)

        # Keep the original XML URI for reference
        provision_uri = element.get(
            "IdURI",
            element.get(
                "{http://www.legislation.gov.uk/namespaces/legislation}IdURI", ""
            ),
        )
        para_el = element.find(f"{{{LEG}}}{level}para")
        text = direct_text(para_el) if para_el is not None else ""

        # is_container: no direct text but has sub-provision children
        sub_children = collect_sub_provisions(para_el) if para_el is not None else []
        is_container = (not text) and len(sub_children) > 0

        full_citation = build_full_citation(ctx)

        return {
            "provision_id": provision_id,
            "node_label": "Provision",
            "xml_id": xml_id,
            "provision_uri": provision_uri,
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
            "provision_number": provision_number,
            "full_citation": full_citation,
            "provision_level": level,
            "parent_provision_id": parent_id,
            "text": text,
            "is_container": is_container,
            "status": element.get("Status", "valid"),
            "restrict_start_date": element.get("RestrictStartDate", ""),
            "restrict_end_date": element.get("RestrictEndDate", ""),
            "restrict_extent": element.get("RestrictExtent", ""),
            "last_parsed": "",  # filled in by main()
        }

    def recurse_sub_provisions(
        para_el, ctx: dict, parent_id: str, parent_level: str, provisions: list
    ):
        """
        Recurse into all sub-provision children of a P*para element.
        Handles skip-level nesting (e.g. P1para -> direct P3).

        ctx["citation_parts"] already contains the FULL path to the parent node.
        Each child extends it by appending its own number.
        parent_id is the constructed provision_id of the parent (passed for ID building).
        """
        if para_el is None:
            return

        for child in para_el:
            child_tag = get_tag(child)
            if child_tag not in SUB_TAGS:
                continue

            child_pn_el = child.find(f"{{{LEG}}}Pnumber")
            child_number = pnumber_text(child_pn_el, child)
            child_xml_id = child.get("id", "")

            child_ctx = dict(ctx)
            child_ctx["citation_parts"] = list(ctx.get("citation_parts", [])) + [
                child_number
            ]

            # Build ID by extending parent_id
            child_provision_id = build_provision_id(
                ctx, child_number, parent_id, child_xml_id
            )

            prov = make_provision(child, child_ctx, child_tag, parent_id)
            # Override the provision_id computed inside make_provision with our correctly-parented one
            prov["provision_id"] = child_provision_id
            prov["full_citation"] = build_full_citation(child_ctx)
            provisions.append(prov)

            # Recurse further into this child's para
            child_para = child.find(f"{{{LEG}}}{child_tag}para")
            recurse_sub_provisions(
                child_para, child_ctx, child_provision_id, child_tag, provisions
            )

    def walk(element, context: dict):
        """Recursively walk the XML tree, building up hierarchy context."""
        tag = get_tag(element)
        ctx = dict(context)

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
            num_el = element.find(f"{{{LEG}}}Number")
            title_el = element.find(f"{{{LEG}}}TitleBlock/{{{LEG}}}Title")
            if title_el is None:
                title_el = element.find(f"{{{LEG}}}TitleBlock")
            ctx["schedule_number"] = text_of(num_el)
            ctx["schedule_title"] = text_of(title_el)
            ctx["schedule_id"] = element.get("id", "")
            # Record structural xml_id -> normalised id
            xml_id = element.get("id", "")
            if xml_id and ctx["schedule_number"]:
                structural_ids[xml_id] = normalise_schedule(ctx["schedule_number"])
            for child in element:
                walk(child, ctx)
            return

        if tag == "Part":
            num_el = element.find(f"{{{LEG}}}Number")
            title_el = element.find(f"{{{LEG}}}Title")
            ctx["part_number"] = text_of(num_el)
            ctx["part_title"] = text_of(title_el)
            ctx["part_id"] = element.get("id", "")
            # Record structural xml_id -> normalised id
            xml_id = element.get("id", "")
            if xml_id and ctx.get("schedule_number") and ctx["part_number"]:
                pt_id = f"{normalise_schedule(ctx['schedule_number'])}-{normalise_part(ctx['part_number'])}"
                structural_ids[xml_id] = pt_id
            for child in element:
                walk(child, ctx)
            return

        if tag == "Chapter":
            num_el = element.find(f"{{{LEG}}}Number")
            title_el = element.find(f"{{{LEG}}}Title")
            ctx["chapter_number"] = text_of(num_el)
            ctx["chapter_title"] = text_of(title_el)
            for child in element:
                walk(child, ctx)
            return

        if tag == "Pblock":
            title_el = element.find(f"{{{LEG}}}Title")
            ctx["class_block"] = text_of(title_el)
            ctx["class_block_id"] = element.get("id", "")
            # Record structural xml_id -> normalised id
            xml_id = element.get("id", "")
            if (
                xml_id
                and ctx.get("schedule_number")
                and ctx.get("part_number")
                and ctx["class_block"]
            ):
                m = re.match(
                    r"Class ([A-Z]+[A-Z0-9]*)", ctx["class_block"], re.IGNORECASE
                )
                if m:
                    cls_id = (
                        f"{normalise_schedule(ctx['schedule_number'])}"
                        f"-{normalise_part(ctx['part_number'])}"
                        f"-cls{m.group(1).upper()}"
                    )
                    structural_ids[xml_id] = cls_id
            for child in element:
                walk(child, ctx)
            return

        if tag == "PsubBlock":
            title_el = element.find(f"{{{LEG}}}Title")
            ctx["sub_block"] = text_of(title_el)
            ctx["sub_block_id"] = element.get("id", "")
            for child in element:
                walk(child, ctx)
            return

        if tag == "P1":
            pn_el = element.find(f"{{{LEG}}}Pnumber")
            provision_number = pnumber_text(pn_el, element)
            xml_id = element.get("id", "")
            p1_id = build_provision_id(ctx, provision_number, "", xml_id)

            # citation_parts starts fresh at P1 level, containing just this provision's number
            p1_ctx = dict(ctx)
            p1_ctx["citation_parts"] = [provision_number]

            p1para = element.find(f"{{{LEG}}}P1para")
            text = direct_text(p1para) if p1para is not None else ""
            sub_children = collect_sub_provisions(p1para) if p1para is not None else []
            is_container = (not text) and len(sub_children) > 0

            full_citation = build_full_citation(p1_ctx)

            provision = {
                "provision_id": p1_id,
                "node_label": "Provision",
                "xml_id": xml_id,
                "provision_uri": element.get(
                    "IdURI",
                    element.get(
                        "{http://www.legislation.gov.uk/namespaces/legislation}IdURI",
                        "",
                    ),
                ),
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
                "provision_number": provision_number,
                "full_citation": full_citation,
                "provision_level": "P1",
                "parent_provision_id": "",
                "text": text,
                "is_container": is_container,
                "status": element.get("Status", "valid"),
                "restrict_start_date": element.get("RestrictStartDate", ""),
                "restrict_end_date": element.get("RestrictEndDate", ""),
                "restrict_extent": element.get("RestrictExtent", ""),
                "last_parsed": "",  # filled in by main()
            }
            provisions.append(provision)

            # Recurse into all sub-provision children (handles skip-level)
            recurse_sub_provisions(p1para, p1_ctx, p1_id, "P1", provisions)
            return

        # For any other element, just recurse
        for child in element:
            walk(child, ctx)

    # Start walking
    leg_el = root.find(".//leg:Body", NAMESPACES)
    if leg_el is not None:
        walk(leg_el, {})

    sched_el = root.find(".//leg:Schedules", NAMESPACES)
    if sched_el is not None:
        walk(sched_el, {})

    return provisions, structural_ids


# ---------------------------------------------------------------------------
# Cross-references
# ---------------------------------------------------------------------------


def parse_cross_references(root, xmlid_to_provid: dict) -> list[dict]:
    """
    Extract all cross-references from the GPDO CLML XML.

    CLML cross-reference structure:
      1. <CommentaryRef> inside provisions (263 total) — a provision footnote marker
         that points to a <Commentary> element by Ref attribute.
      2. <Citation> inside <Commentary> elements (790 total) — the footnote text
         itself, citing the amending or enabling legislation.

    Resolution strategy:
      - CommentaryRef rows: source_provision_id is the enclosing provision (direct lookup).
      - Citation rows: source_provision_id is resolved by finding all provisions
        that carry a <CommentaryRef> pointing to this Commentary, i.e. we trace
        the chain  Provision -> CommentaryRef -> Commentary -> Citation -> Legislation.
        This correctly assigns each citation to the provision(s) it relates to.
    """
    LEG = "http://www.legislation.gov.uk/namespaces/legislation"
    refs = []
    counter = [0]

    def nearest_provid(element) -> str:
        """
        Walk up the tree to find the nearest ancestor that maps to a provision_id.
        Falls back to LEGISLATION_ID for preamble-level elements.
        """
        parent = element.getparent()
        while parent is not None:
            pid = parent.get("id", "")
            if pid and pid in xmlid_to_provid:
                return xmlid_to_provid[pid]
            tag = etree.QName(parent.tag).localname
            # Preamble containers — assign to legislation root
            if tag in ("SecondaryPrelims", "PrimaryPrelims", "Schedules", "Body"):
                return LEGISLATION_ID
            parent = parent.getparent()
        return LEGISLATION_ID  # ultimate fallback

    # --- Step 1: Build commentary_id -> [provision_ids] reverse map ---
    # Walk every CommentaryRef in the document; record which provision owns it.
    commentary_to_provisions: dict[str, list[str]] = {}
    for cref in root.iter(f"{{{LEG}}}CommentaryRef"):
        comm_id = cref.get("Ref", "")
        if not comm_id:
            continue
        prov_id = nearest_provid(cref)
        if prov_id:
            commentary_to_provisions.setdefault(comm_id, [])
            if prov_id not in commentary_to_provisions[comm_id]:
                commentary_to_provisions[comm_id].append(prov_id)

    # --- Step 2: CommentaryRef rows (one per provision-footnote link) ---
    for cref in root.iter(f"{{{LEG}}}CommentaryRef"):
        comm_id = cref.get("Ref", "")
        source_id = nearest_provid(cref)
        counter[0] += 1
        refs.append(
            {
                "ref_id": f"cref-{counter[0]:05d}",
                "source_provision_id": source_id,
                "target_uri": "",
                "target_legislation_id": "",
                "target_provision_path": "",
                "citation_text": f"Commentary: {comm_id}",
                "is_internal": True,
                "reference_type": "commentary",
            }
        )

    # --- Step 3: Citation rows, resolved via Commentary chain ---
    for citation in root.iter(f"{{{LEG}}}Citation"):
        target_uri = citation.get("URI", "")
        citation_text = text_of(citation)

        is_internal = LEGISLATION_ID in target_uri if target_uri else False

        target_legislation = ""
        target_provision = ""
        if target_uri:
            clean = target_uri.replace("http://www.legislation.gov.uk/id/", "")
            parts = clean.split("/")
            if len(parts) >= 3:
                target_legislation = "/".join(parts[:3])
                if len(parts) > 3:
                    target_provision = "/".join(parts[3:])

        # Determine source provision(s)
        # Try direct ancestor lookup first (handles any future inline citations)
        direct_source = nearest_provid(citation)
        if direct_source:
            sources = [direct_source]
        else:
            # Citation is inside a Commentary element — resolve via reverse map
            sources = []
            parent = citation.getparent()
            while parent is not None:
                tag = etree.QName(parent.tag).localname
                if tag == "Commentary":
                    comm_id = parent.get("id", "")
                    sources = commentary_to_provisions.get(comm_id, [])
                    break
                parent = parent.getparent()

        if not sources:
            # Citation has no resolvable provision (e.g. preamble / enabling powers)
            sources = [""]

        for source_id in sources:
            counter[0] += 1
            refs.append(
                {
                    "ref_id": f"xref-{counter[0]:05d}",
                    "source_provision_id": source_id,
                    "target_uri": target_uri,
                    "target_legislation_id": target_legislation,
                    "target_provision_path": target_provision,
                    "citation_text": citation_text,
                    "is_internal": is_internal,
                    "reference_type": (
                        "amendment_source" if not direct_source else "citation"
                    ),
                }
            )

    return refs


# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------


def parse_definitions(root, xmlid_to_provid: dict) -> list[dict]:
    """
    Extract defined terms from <Term> elements.
    Uses xmlid_to_provid to resolve source_provision_id to our stable provision_id.
    """
    LEG = "http://www.legislation.gov.uk/namespaces/legislation"
    definitions = []

    for i, term_el in enumerate(root.iter(f"{{{LEG}}}Term")):
        term_text = text_of(term_el)
        term_id = term_el.get("id", "")

        source_id = ""
        parent = term_el.getparent()
        while parent is not None:
            pid = parent.get("id", "")
            if pid and pid in xmlid_to_provid:
                source_id = xmlid_to_provid[pid]
                break
            parent = parent.getparent()

        def_text = ""
        text_parent = term_el.getparent()
        if text_parent is not None:
            def_text = text_of(text_parent)

        definitions.append(
            {
                "def_id": f"def-{i+1:05d}",
                "term": term_text,
                "term_id": term_id,
                "definition_text": def_text,
                "source_provision_id": source_id,
                "legislation_id": LEGISLATION_ID,
            }
        )

    return definitions


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


def build_hierarchy(provisions: list[dict]) -> list[dict]:
    """
    Build explicit parent-child edges for the graph.

    Fixes:
    - Body articles now correctly link to the legislation root (not a null schedule)
    - All structural node IDs (schedule/part/class) use the same normalised form
      as the structural node rows in articles.csv, so every edge endpoint resolves.
    - Each edge has a unique edge_id PK.
    """
    hierarchy = []
    seen = set()
    edge_counter = [0]

    def add_edge(
        parent_type,
        parent_id,
        parent_label,
        relationship,
        child_type,
        child_id,
        child_label,
    ):
        key = (parent_id, relationship, child_id)
        if key in seen or not child_id:
            return
        seen.add(key)
        edge_counter[0] += 1
        hierarchy.append(
            {
                "edge_id": f"edge-{edge_counter[0]:05d}",
                "parent_type": parent_type,
                "parent_id": parent_id,
                "parent_label": parent_label,
                "relationship": relationship,
                "child_type": child_type,
                "child_id": child_id,
                "child_label": child_label,
            }
        )

    for prov in provisions:
        section_type = prov.get("section_type", "")
        sch_num = prov.get("schedule_number", "")
        pt_num = prov.get("part_number", "")
        cls_block = prov.get("class_block", "")

        # Normalised structural IDs (match what build_structural_nodes() produces)
        sch_id = normalise_schedule(sch_num) if sch_num else ""
        pt_id = f"{sch_id}-{normalise_part(pt_num)}" if (sch_id and pt_num) else ""
        cls_id = ""
        if sch_id and pt_id and cls_block:
            m = re.match(r"Class ([A-Z]+[A-Z0-9]*)", cls_block, re.IGNORECASE)
            if m:
                cls_id = f"{pt_id}-cls{m.group(1).upper()}"

        # --- Body articles: link to legislation root ---
        if section_type == "body" and prov["provision_level"] == "P1":
            add_edge(
                "legislation",
                LEGISLATION_ID,
                LEGISLATION_SHORT,
                "has_article",
                "provision",
                prov["provision_id"],
                prov["full_citation"],
            )

        # --- Legislation -> Schedule ---
        if sch_num and sch_id:
            add_edge(
                "legislation",
                LEGISLATION_ID,
                LEGISLATION_SHORT,
                "has_schedule",
                "schedule",
                sch_id,
                f"{sch_num} – {prov['schedule_title']}",
            )

        # --- Schedule -> Part ---
        if sch_id and pt_id:
            add_edge(
                "schedule",
                sch_id,
                sch_num,
                "has_part",
                "part",
                pt_id,
                f"{pt_num} – {prov['part_title']}",
            )

        # --- Part -> Class block ---
        if pt_id and cls_id:
            add_edge("part", pt_id, pt_num, "has_class", "class", cls_id, cls_block)

        # --- Class / Part / Schedule -> P1 provision ---
        if prov["provision_level"] == "P1" and section_type != "body":
            if cls_id:
                add_edge(
                    "class",
                    cls_id,
                    cls_block,
                    "has_provision",
                    "provision",
                    prov["provision_id"],
                    prov["full_citation"],
                )
            elif pt_id:
                add_edge(
                    "part",
                    pt_id,
                    pt_num,
                    "has_provision",
                    "provision",
                    prov["provision_id"],
                    prov["full_citation"],
                )
            elif sch_id:
                add_edge(
                    "schedule",
                    sch_id,
                    sch_num,
                    "has_provision",
                    "provision",
                    prov["provision_id"],
                    prov["full_citation"],
                )

        # --- P1 -> P2/P3/P4/P5 (sub-provisions at any depth) ---
        if prov["parent_provision_id"] and prov["provision_level"] in (
            "P2",
            "P3",
            "P4",
            "P5",
        ):
            add_edge(
                "provision",
                prov["parent_provision_id"],
                prov["parent_provision_id"],
                "has_sub_provision",
                "provision",
                prov["provision_id"],
                prov["full_citation"],
            )

    return hierarchy


# ---------------------------------------------------------------------------
# ID mapping helper
# ---------------------------------------------------------------------------


def build_id_to_shortid(root) -> dict:
    """
    Build a dict mapping every full XML id -> shortId across the document.
    Used to normalise cross-reference and definition source IDs.
    """
    mapping = {}
    LEG = "http://www.legislation.gov.uk/namespaces/legislation"
    for el in root.iter():
        full_id = el.get("id", "")
        short_id = el.get("shortId", "")
        if full_id and short_id:
            mapping[full_id] = short_id
    return mapping


def build_xmlid_to_provid(provisions: list[dict]) -> dict:
    """
    Build a dict mapping xml_id -> provision_id from the parsed provisions.

    This is the critical lookup that fixes cross-references and definitions:
    the nearest-ancestor XML id found in the CLML tree maps to our stable
    legal-citation-based provision_id (e.g. "art-3-1-a"), not to the stale
    shortId that was used before.
    """
    mapping = {}
    for prov in provisions:
        xml_id = prov.get("xml_id", "")
        provision_id = prov.get("provision_id", "")
        if xml_id and provision_id:
            mapping[xml_id] = provision_id
    return mapping


def parse_legislation_id(legislation_id: str) -> tuple[str, str, str]:
    """Split legislation id like 'ukpga/1990/8' into (type, year, number)."""
    parts = (legislation_id or "").split("/")
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    return "", "", ""


def fetch_legislation_metadata(legislation_id: str) -> dict:
    """
    Fetch lightweight metadata for a legislation id from legislation.gov.uk.
    Returns a dict with title, document_type, year, number and uri.
    """
    doc_type, year, number = parse_legislation_id(legislation_id)
    # Keep canonical URI in /id form for graph nodes.
    uri = f"https://www.legislation.gov.uk/id/{legislation_id}"
    canonical_uri = resolve_canonical_legislation_uri(legislation_id)
    cache_base = legislation_id.replace("/", "_")

    # Deterministic XML metadata fetch against canonical URI.
    xml_meta = {}
    xml_candidates = [
        (
            f"{canonical_uri}/resources/data.xml",
            CACHE_DIR / f"{cache_base}_resources_data.xml",
        ),
        (
            f"{canonical_uri}/data.xml",
            CACHE_DIR / f"{cache_base}_data.xml",
        ),
    ]
    for x_url, x_cache in xml_candidates:
        xml_meta = download_xml_metadata(x_url, x_cache, verbose=False)
        if xml_meta:
            break

    # Use XML metadata first, then ID-derived fallback.
    title = str(xml_meta.get("title", "")).strip()
    if not title:
        title = legislation_id

    document_type = (
        str(xml_meta.get("document_type", "")).strip()
        or doc_type
    )
    year_val = (
        str(xml_meta.get("year", "")).strip()
        or year
    )
    number_val = (
        str(xml_meta.get("number", "")).strip()
        or number
    )

    return {
        "id": legislation_id,
        "title": title,
        "document_type": document_type,
        "year": year_val,
        "number": number_val,
        "uri": uri,
    }


def namespace_graph_ids(
    legislation_id: str, nodes: list[dict], hierarchy: list[dict]
) -> tuple[list[dict], list[dict]]:
    """
    Namespace all non-root node IDs for a document to avoid collisions when
    multiple legislations are merged into one graph.
    """
    id_map: dict[str, str] = {}
    for n in nodes:
        pid = n.get("provision_id", "")
        if pid and pid != legislation_id:
            id_map[pid] = f"{legislation_id}::{pid}"

    for n in nodes:
        old_id = n.get("provision_id", "")
        parent_id = n.get("parent_provision_id", "")
        if old_id in id_map:
            n["provision_id"] = id_map[old_id]
        if parent_id in id_map:
            n["parent_provision_id"] = id_map[parent_id]

    for e in hierarchy:
        p = e.get("parent_id", "")
        c = e.get("child_id", "")
        if p in id_map:
            e["parent_id"] = id_map[p]
        if c in id_map:
            e["child_id"] = id_map[c]

    return nodes, hierarchy


def reindex_edge_ids(hierarchy: list[dict]) -> list[dict]:
    """Ensure edge_id values are globally unique after hierarchy merges."""
    for i, edge in enumerate(hierarchy, start=1):
        edge["edge_id"] = f"edge-{i:05d}"
    return hierarchy


def build_external_stub_nodes(
    cross_refs: list[dict], existing_node_ids: set[str], parsed_at: str
) -> list[dict]:
    """Create one lightweight node for each externally referenced legislation."""
    external_leg_ids = sorted(
        {
            r.get("target_legislation_id", "")
            for r in cross_refs
            if r.get("target_legislation_id", "")
            and r.get("target_legislation_id", "") != LEGISLATION_ID
        }
    )

    pending_leg_ids = [leg_id for leg_id in external_leg_ids if leg_id not in existing_node_ids]
    total = len(pending_leg_ids)
    if total == 0:
        return []

    print(f"  Resolving metadata for {total} external legislations...")
    metadata_by_id: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=METADATA_FETCH_WORKERS) as executor:
        future_map = {
            executor.submit(fetch_legislation_metadata, leg_id): leg_id
            for leg_id in pending_leg_ids
        }
        for i, future in enumerate(as_completed(future_map), start=1):
            leg_id = future_map[future]
            try:
                metadata_by_id[leg_id] = future.result()
            except Exception:
                doc_type, year, number = parse_legislation_id(leg_id)
                metadata_by_id[leg_id] = {
                    "id": leg_id,
                    "title": leg_id,
                    "document_type": doc_type,
                    "year": year,
                    "number": number,
                    "uri": f"https://www.legislation.gov.uk/id/{leg_id}",
                }
            if i == 1 or i % 10 == 0 or i == total:
                print(f"    metadata progress: {i}/{total}")

    nodes: list[dict] = []
    for leg_id in pending_leg_ids:
        md = metadata_by_id[leg_id]
        node_text = f"{md['title']} ({md['document_type']} {md['year']}/{md['number']})"
        nodes.append(
            {
                "provision_id": leg_id,
                "node_label": "Legislation",
                "xml_id": "",
                "provision_uri": md["uri"],
                "legislation_id": leg_id,
                "legislation_title": md["title"],
                "section_type": "",
                "schedule_number": "",
                "schedule_title": "",
                "part_number": "",
                "part_title": "",
                "chapter_number": "",
                "chapter_title": "",
                "class_block": "",
                "sub_block": "",
                "provision_number": md["number"],
                "full_citation": md["title"],
                "provision_level": "",
                "parent_provision_id": "",
                "text": node_text,
                "is_container": True,
                "status": "valid",
                "restrict_start_date": "",
                "restrict_end_date": "",
                "restrict_extent": "",
                "last_parsed": parsed_at,
            }
        )
    return nodes


def parse_full_legislation(
    xml_bytes: bytes, legislation_id: str, legislation_title: str, legislation_short: str
) -> tuple[list[dict], list[dict]]:
    """
    Parse a full legislation document with the current parser stack and return:
      (all_nodes_for_document, hierarchy_edges_for_document)
    """
    global LEGISLATION_ID, LEGISLATION_TITLE, LEGISLATION_SHORT

    old_ctx = (LEGISLATION_ID, LEGISLATION_TITLE, LEGISLATION_SHORT)
    parsed_at = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    try:
        LEGISLATION_ID = legislation_id
        LEGISLATION_TITLE = legislation_title
        LEGISLATION_SHORT = legislation_short

        root = etree.fromstring(xml_bytes)
        provisions, _ = parse_provisions(root)
        for prov in provisions:
            prov["last_parsed"] = parsed_at
        structural_nodes = build_structural_nodes(provisions, parsed_at)
        hierarchy = build_hierarchy(provisions)
        all_nodes = structural_nodes + provisions
        return all_nodes, hierarchy
    finally:
        LEGISLATION_ID, LEGISLATION_TITLE, LEGISLATION_SHORT = old_ctx


def build_structural_nodes(provisions: list[dict], parsed_at: str) -> list[dict]:
    """
    Generate explicit node rows for structural containers:
    Legislation, Schedule, Part, and Class.

    These are the nodes that appear as parents in hierarchy.csv but previously
    had no corresponding row in articles.csv. Adding them ensures every node
    referenced in an edge actually exists in the node table.
    """
    nodes = []
    seen = set()

    # Legislation root node
    leg_id = LEGISLATION_ID
    if leg_id not in seen:
        seen.add(leg_id)
        _, leg_year, leg_number = parse_legislation_id(leg_id)
        leg_doc_number = (
            f"{leg_year}/{leg_number}" if (leg_year and leg_number) else leg_id
        )
        nodes.append(
            {
                "provision_id": leg_id,
                "node_label": "Legislation",
                "xml_id": "",
                "provision_uri": f"https://www.legislation.gov.uk/id/{leg_id}",
                "legislation_id": leg_id,
                "legislation_title": LEGISLATION_SHORT,
                "section_type": "",
                "schedule_number": "",
                "schedule_title": "",
                "part_number": "",
                "part_title": "",
                "chapter_number": "",
                "chapter_title": "",
                "class_block": "",
                "sub_block": "",
                "provision_number": leg_doc_number,
                "full_citation": LEGISLATION_TITLE,
                "provision_level": "",
                "parent_provision_id": "",
                "text": LEGISLATION_TITLE,
                "is_container": True,
                "status": "valid",
                "restrict_start_date": "",
                "restrict_end_date": "",
                "restrict_extent": "E+W",
                "last_parsed": parsed_at,
            }
        )

    for prov in provisions:
        sch_num = prov.get("schedule_number", "")
        sch_title = prov.get("schedule_title", "")
        pt_num = prov.get("part_number", "")
        pt_title = prov.get("part_title", "")
        cls_block = prov.get("class_block", "")

        # Schedule node
        if sch_num:
            sch_id = normalise_schedule(sch_num)
            if sch_id not in seen:
                seen.add(sch_id)
                nodes.append(
                    {
                        "provision_id": sch_id,
                        "node_label": "Schedule",
                        "xml_id": "",
                        "provision_uri": "",
                        "legislation_id": LEGISLATION_ID,
                        "legislation_title": LEGISLATION_SHORT,
                        "section_type": "schedules",
                        "schedule_number": sch_num,
                        "schedule_title": sch_title,
                        "part_number": "",
                        "part_title": "",
                        "chapter_number": "",
                        "chapter_title": "",
                        "class_block": "",
                        "sub_block": "",
                        "provision_number": sch_num,
                        "full_citation": sch_num.title(),
                        "provision_level": "",
                        "parent_provision_id": LEGISLATION_ID,
                        "text": f"{sch_num} – {sch_title}",
                        "is_container": True,
                        "status": "valid",
                        "restrict_start_date": "",
                        "restrict_end_date": "",
                        "restrict_extent": "E+W",
                        "last_parsed": parsed_at,
                    }
                )

        # Part node (unique within its schedule)
        if sch_num and pt_num:
            pt_id = f"{normalise_schedule(sch_num)}-{normalise_part(pt_num)}"
            if pt_id not in seen:
                seen.add(pt_id)
                nodes.append(
                    {
                        "provision_id": pt_id,
                        "node_label": "Part",
                        "xml_id": "",
                        "provision_uri": "",
                        "legislation_id": LEGISLATION_ID,
                        "legislation_title": LEGISLATION_SHORT,
                        "section_type": "schedules",
                        "schedule_number": sch_num,
                        "schedule_title": sch_title,
                        "part_number": pt_num,
                        "part_title": pt_title,
                        "chapter_number": "",
                        "chapter_title": "",
                        "class_block": "",
                        "sub_block": "",
                        "provision_number": pt_num,
                        "full_citation": f"{sch_num.title()}, {pt_num.title()}",
                        "provision_level": "",
                        "parent_provision_id": normalise_schedule(sch_num),
                        "text": f"{pt_num} – {pt_title}",
                        "is_container": True,
                        "status": "valid",
                        "restrict_start_date": "",
                        "restrict_end_date": "",
                        "restrict_extent": "E+W",
                        "last_parsed": parsed_at,
                    }
                )

        # Class node (unique within its part+schedule)
        if sch_num and pt_num and cls_block:
            m = re.match(r"Class ([A-Z]+[A-Z0-9]*)", cls_block, re.IGNORECASE)
            cls_code = m.group(1).upper() if m else ""
            if cls_code:
                cls_id = f"{normalise_schedule(sch_num)}-{normalise_part(pt_num)}-cls{cls_code}"
                if cls_id not in seen:
                    seen.add(cls_id)
                    nodes.append(
                        {
                            "provision_id": cls_id,
                            "node_label": "Class",
                            "xml_id": "",
                            "provision_uri": "",
                            "legislation_id": LEGISLATION_ID,
                            "legislation_title": LEGISLATION_SHORT,
                            "section_type": "schedules",
                            "schedule_number": sch_num,
                            "schedule_title": sch_title,
                            "part_number": pt_num,
                            "part_title": pt_title,
                            "chapter_number": "",
                            "chapter_title": "",
                            "class_block": cls_block,
                            "sub_block": "",
                            "provision_number": cls_code,
                            "full_citation": f"{sch_num.title()}, {pt_num.title()}, Class {cls_code}",
                            "provision_level": "",
                            "parent_provision_id": f"{normalise_schedule(sch_num)}-{normalise_part(pt_num)}",
                            "text": cls_block,
                            "is_container": True,
                            "status": "valid",
                            "restrict_start_date": "",
                            "restrict_end_date": "",
                            "restrict_extent": "E+W",
                            "last_parsed": parsed_at,
                        }
                    )

    return nodes


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def parse_metadata(root) -> dict:
    meta = {}
    metadata_el = root.find(".//ukm:Metadata", NAMESPACES)
    if metadata_el is not None:
        title_el = metadata_el.find(".//dc:title", NAMESPACES)
        if title_el is not None:
            meta["title"] = text_of(title_el)
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
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Capture parse timestamp once for the whole run
    parsed_at = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    # 1. Download / load
    xml_bytes = download_xml(GPDO_URL)

    # 2. Parse XML
    print("Parsing XML...")
    root = etree.fromstring(xml_bytes)

    # 3. (Legacy mapping kept for reference; cross-refs now use xmlid_to_provid)
    print("Building ID mapping...")
    id_to_shortid = build_id_to_shortid(root)
    print(f"  {len(id_to_shortid)} id->shortId mappings found")

    # 4. Metadata
    meta = parse_metadata(root)
    print(f"  Document: {meta.get('title', 'unknown')}")
    print(
        f"  Type: {meta.get('document_type', '?')}, Year: {meta.get('year', '?')}, Number: {meta.get('number', '?')}"
    )

    # 5. Provisions
    print("Extracting provisions...")
    provisions, structural_ids = parse_provisions(root)
    print(f"  Found {len(provisions)} provisions")

    # 6. Stamp last_parsed on all provisions and build xml_id lookup
    for prov in provisions:
        prov["last_parsed"] = parsed_at

    xmlid_to_provid = build_xmlid_to_provid(provisions)
    # Merge structural element ids (Schedule/Part/Class XML ids) so citations
    # inside those containers also resolve to their parent structural node id
    xmlid_to_provid.update(structural_ids)
    print(
        f"  Built xml_id -> provision_id lookup: {len(xmlid_to_provid)} entries "
        f"({len(structural_ids)} structural)"
    )

    # 7. Cross-references (now uses xmlid_to_provid for correct source IDs)
    print("Extracting cross-references...")
    cross_refs = parse_cross_references(root, xmlid_to_provid)
    print(f"  Found {len(cross_refs)} cross-references")

    # Diagnostic: how many cross-refs resolved to a known provision_id?
    # Note: check against all_nodes (including structural) not just provisions
    # This is evaluated after all_nodes is built below, so we defer to a lambda
    _cross_refs = cross_refs  # reference for later diagnostic

    # 8. Definitions (now uses xmlid_to_provid for correct source IDs)
    print("Extracting definitions...")
    definitions = parse_definitions(root, xmlid_to_provid)
    print(f"  Found {len(definitions)} defined terms")

    # 9. Structural nodes (Legislation / Schedule / Part / Class)
    print("Building structural nodes...")
    structural_nodes = build_structural_nodes(provisions, parsed_at)
    print(f"  Found {len(structural_nodes)} structural nodes")

    # 10. Hierarchy (now uses normalised structural IDs)
    print("Building hierarchy...")
    hierarchy = build_hierarchy(provisions)
    print(f"  Found {len(hierarchy)} hierarchy edges")

    # 11. Full parse for key external Act: TCPA 1990 (ukpga/1990/8)
    print(f"Parsing key external Act in full: {TCPA_ID} ...")
    tcpa_nodes: list[dict] = []
    tcpa_hierarchy: list[dict] = []
    try:
        tcpa_xml = download_xml(TCPA_URL, CACHE_DIR / "ukpga_1990_8.xml")
        tcpa_title = "Town and Country Planning Act 1990"
        tcpa_short = "TCPA 1990"
        tcpa_nodes, tcpa_hierarchy = parse_full_legislation(
            tcpa_xml, TCPA_ID, tcpa_title, tcpa_short
        )
        tcpa_nodes, tcpa_hierarchy = namespace_graph_ids(
            TCPA_ID, tcpa_nodes, tcpa_hierarchy
        )
        print(
            f"  Parsed TCPA: {len(tcpa_nodes)} nodes, {len(tcpa_hierarchy)} hierarchy edges"
        )
    except Exception as e:
        print(f"  Warning: TCPA full parse failed, continuing with stubs only: {e}")

    # 12. External stub nodes for all referenced legislation (except GPDO)
    print("Building external legislation stubs...")
    gpdo_node_ids = {n["provision_id"] for n in (structural_nodes + provisions)}
    gpdo_node_ids.update(n.get("provision_id", "") for n in tcpa_nodes)
    external_stub_nodes = build_external_stub_nodes(cross_refs, gpdo_node_ids, parsed_at)
    print(f"  Built {len(external_stub_nodes)} external legislation stubs")

    # 13. Save CSVs
    print("Saving CSVs...")

    # articles.csv: GPDO + TCPA(full) + external stubs, xml_id dropped (internal only)
    all_nodes = structural_nodes + provisions + tcpa_nodes + external_stub_nodes
    all_hierarchy = reindex_edge_ids(hierarchy + tcpa_hierarchy)
    df_provisions = pd.DataFrame(all_nodes)

    # Resolution diagnostics (now that we have the full node id set)
    all_node_ids = {n["provision_id"] for n in all_nodes}
    xref_resolved = sum(
        1 for r in cross_refs if r["source_provision_id"] in all_node_ids
    )
    print(
        f"  Cross-ref source resolution: {xref_resolved}/{len(cross_refs)} (100% expected)"
    )
    def_resolved = sum(
        1 for d in definitions if d["source_provision_id"] in all_node_ids
    )
    print(
        f"  Definition source resolution: {def_resolved}/{len(definitions)} (100% expected)"
    )
    # Drop internal xml_id column (used only during parsing for the lookup)
    if "xml_id" in df_provisions.columns:
        df_provisions = df_provisions.drop(columns=["xml_id"])
    # Ensure provision_id is first column
    cols = ["provision_id"] + [c for c in df_provisions.columns if c != "provision_id"]
    df_provisions = df_provisions[cols]
    df_provisions.to_csv(OUTPUT_DIR / "articles.csv", index=False)
    print(
        f"  articles.csv: {len(df_provisions)} rows, {len(df_provisions.columns)} columns"
    )

    # hierarchy.csv: edge_id first
    df_hier = pd.DataFrame(all_hierarchy)
    df_hier.to_csv(OUTPUT_DIR / "hierarchy.csv", index=False)
    print(f"  hierarchy.csv: {len(df_hier)} rows")

    # cross_references.csv: ref_id first
    df_refs = pd.DataFrame(cross_refs)
    df_refs.to_csv(OUTPUT_DIR / "cross_references.csv", index=False)
    print(f"  cross_references.csv: {len(df_refs)} rows")

    # definitions.csv: def_id first
    df_defs = pd.DataFrame(definitions)
    df_defs.to_csv(OUTPUT_DIR / "definitions.csv", index=False)
    print(f"  definitions.csv: {len(df_defs)} rows")

    # 12. Summary
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUT")
    print("=" * 60)

    provisions_only = df_provisions[df_provisions["node_label"] == "Provision"]
    print(f"\nNode counts by label:")
    print(df_provisions["node_label"].value_counts().to_string())

    if len(provisions_only) > 0:
        print("\n--- Body articles sample ---")
        body = provisions_only[provisions_only["section_type"] == "body"]
        print(
            body[
                [
                    "provision_id",
                    "provision_number",
                    "provision_level",
                    "full_citation",
                    "is_container",
                    "text",
                ]
            ]
            .head(6)
            .to_string(max_colwidth=80)
        )

    print("\n--- Cross-reference source resolution sample ---")
    print(
        df_refs[
            ["ref_id", "source_provision_id", "target_legislation_id", "citation_text"]
        ]
        .head(8)
        .to_string(max_colwidth=60)
    )

    if len(df_hier) > 0:
        print("\n--- Relationship counts ---")
        print(df_hier["relationship"].value_counts().to_string())

    print(f"\nAll files saved to: {OUTPUT_DIR.resolve()}")
    print("Done!")


if __name__ == "__main__":
    main()
