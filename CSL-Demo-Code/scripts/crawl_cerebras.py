#!/usr/bin/env python3
"""
Cerebras SDK Documentation Crawler

Crawls https://sdk.cerebras.net/ and saves all pages as Markdown files
with directory structure mirroring the URL paths. Generates an index.md
as a table of contents with relative links.

Requirements:
    pip install requests beautifulsoup4 markdownify

Usage:
    python crawl_cerebras.py                  # Full crawl
    python crawl_cerebras.py --max-pages 10   # Quick test with 10 pages
    python crawl_cerebras.py --resume         # Skip already-downloaded pages
"""

import argparse
import os
import re
import sys
import time
from collections import deque
from datetime import datetime
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup

try:
    from markdownify import MarkdownConverter
except ImportError:
    print("ERROR: markdownify is required. Install it with:")
    print("  pip install markdownify")
    sys.exit(1)


# ─── Configuration ────────────────────────────────────────────────────────────

START_URL = "https://sdk.cerebras.net/"
DOMAIN = "sdk.cerebras.net"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Cerebras_Docs")

DEFAULT_MAX_DEPTH = 5
DEFAULT_MAX_PAGES = 500
REQUEST_DELAY = 0.3          # Seconds between requests (be polite)
REQUEST_TIMEOUT = 30

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# CSS selectors to find the main content area (tried in order)
CONTENT_SELECTORS = [
    "[role='main']",
    "main",
    "article",
    ".body",
    ".content",
    "#content",
]

# Elements to strip from the content before conversion
REMOVE_SELECTORS = [
    "nav",
    "footer",
    "header",
    "script",
    "style",
    "noscript",
    ".headerlink",          # Sphinx ¶ permalink anchors
    ".sidebar",
    ".sphinxsidebar",
    ".related",
    ".breadcrumb",
    ".pagination",
    ".edit-this-page",
    ".rst-footer-buttons",
]

# File extensions to skip when discovering links
SKIP_EXTENSIONS = {
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".css", ".js", ".json", ".xml",
    ".woff", ".woff2", ".ttf", ".eot",
    ".mp4", ".mp3", ".wav",
}


# ─── Custom Markdown Converter ────────────────────────────────────────────────

class CerebrasMarkdownConverter(MarkdownConverter):
    """Custom converter that handles Sphinx-specific HTML patterns."""

    def convert_pre(self, el, text, parent_tags=None):
        """Preserve code blocks with language hints."""
        code = el.find("code") or el.find("span")
        lang = ""
        if code:
            classes = code.get("class", [])
            for cls in classes:
                if cls.startswith("highlight-") or cls.startswith("language-"):
                    lang = cls.split("-", 1)[1]
                    break
            # Also check parent div for Sphinx highlight class
            parent = el.parent
            if not lang and parent and parent.name == "div":
                for cls in parent.get("class", []):
                    if cls.startswith("highlight-"):
                        lang = cls.split("-", 1)[1]
                        break
        if not lang:
            # Try to detect from the highlight div wrapper
            wrapper = el.find_parent("div", class_=re.compile(r"highlight-"))
            if wrapper:
                for cls in wrapper.get("class", []):
                    if cls.startswith("highlight-"):
                        lang = cls.split("-", 1)[1]
                        break
        # Map common Sphinx language names
        lang_map = {"csl": "csl", "python3": "python", "console": "bash", "none": ""}
        lang = lang_map.get(lang, lang)

        code_text = el.get_text()
        return f"\n```{lang}\n{code_text}\n```\n"

    def convert_table(self, el, text, parent_tags=None):
        """Convert HTML tables to markdown tables."""
        rows = el.find_all("tr")
        if not rows:
            return text

        md_rows = []
        for row in rows:
            cells = row.find_all(["td", "th"])
            cell_texts = []
            for cell in cells:
                cell_text = self.convert(str(cell)).strip()
                # Collapse whitespace within cells
                cell_text = re.sub(r"\s+", " ", cell_text)
                cell_texts.append(cell_text)
            md_rows.append("| " + " | ".join(cell_texts) + " |")

        if len(md_rows) >= 1:
            # Insert separator after first row (header)
            num_cols = md_rows[0].count("|") - 1
            separator = "| " + " | ".join(["---"] * max(num_cols, 1)) + " |"
            md_rows.insert(1, separator)

        return "\n" + "\n".join(md_rows) + "\n"

    def convert_div(self, el, text, parent_tags=None):
        """Handle Sphinx admonitions (note, warning, etc.)."""
        classes = el.get("class", [])

        # Sphinx admonition boxes
        admonition_types = {"note", "warning", "tip", "important", "danger", "caution", "hint"}
        for cls in classes:
            if cls in admonition_types:
                # Extract title from the already-converted text
                # The admonition-title is typically the first line
                body = text.strip()
                lines = body.split("\n")
                # Try to find and separate the title
                title = cls.upper()
                title_el = el.find(class_="admonition-title")
                if title_el:
                    title = title_el.get_text(strip=True)
                quoted = "\n".join(f"> {line}" for line in lines if line.strip())
                return f"\n> **{title}**\n{quoted}\n"

        return text


def html_to_markdown(html_content, base_url):
    """Convert an HTML page to clean Markdown, extracting main content."""
    soup = BeautifulSoup(html_content, "html.parser")

    # Find main content
    content = None
    for selector in CONTENT_SELECTORS:
        content = soup.select_one(selector)
        if content:
            break
    if not content:
        content = soup.body or soup

    # Remove unwanted elements
    for selector in REMOVE_SELECTORS:
        for el in content.select(selector):
            el.decompose()

    # Remove Sphinx source file links (.rst, .pdf download links)
    for a in content.find_all("a", href=True):
        href = a["href"]
        if "_sources/" in href or href.endswith(".rst.txt"):
            a.decompose()

    # Remove floating .pdf download text
    for el in content.find_all(string=re.compile(r"^\s*\.pdf\s*$")):
        if el.parent and el.parent.name in ("a", "li", "p", "span", "div"):
            el.parent.decompose()
        else:
            el.extract()

    # Remove the on-page "Contents" sidebar/TOC (Sphinx local TOC)
    for el in content.select(".contents, .topic.contents, #contents"):
        el.decompose()

    # Convert internal links to relative paths
    for a in content.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/"):
            full = urljoin(base_url, href)
            parsed = urlparse(full)
            if parsed.netloc == DOMAIN:
                # Convert to relative .md path
                path = parsed.path.strip("/")
                if path:
                    md_path = path + ".md"
                else:
                    md_path = "index.md"
                # We'll adjust relative path later; for now keep the .md path
                a["href"] = md_path
                if parsed.fragment:
                    a["href"] += "#" + parsed.fragment

    converter = CerebrasMarkdownConverter(
        heading_style="ATX",
        bullets="-",
        strong_em_symbol="*",
    )
    markdown = converter.convert(str(content))

    # Clean up excessive blank lines
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    # Clean up spaces before newlines
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)
    # Remove leftover .rst/.pdf link lines
    markdown = re.sub(r"^\[\.rst\]\([^)]*\)\s*$", "", markdown, flags=re.MULTILINE)
    markdown = re.sub(r"^\.pdf\s*$", "", markdown, flags=re.MULTILINE)
    # Remove standalone "Contents" lines that are leftover TOC headers
    markdown = re.sub(r"^ Contents\s*$", "", markdown, flags=re.MULTILINE)
    # Remove duplicate consecutive identical headings
    markdown = re.sub(r"(^#{1,6} .+\n)\n*\1", r"\1", markdown, flags=re.MULTILINE)
    # Remove "## Contents" section headers (Sphinx local TOC)
    markdown = re.sub(r"^## Contents\s*$", "", markdown, flags=re.MULTILINE)
    # Final cleanup of excessive blank lines after removals
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip()


# ─── URL / File Path Helpers ──────────────────────────────────────────────────

def url_to_filepath(url):
    """Convert a URL to a local Markdown file path (relative to OUTPUT_DIR)."""
    parsed = urlparse(url)
    path = parsed.path.strip("/")

    if not path:
        return "index.md"

    # Sanitize each path component
    parts = path.split("/")
    parts = [re.sub(r'[<>:"|?*\\]', "-", p) for p in parts if p]

    return os.path.join(*parts) + ".md"


def get_page_title(soup):
    """Extract a clean page title from BeautifulSoup."""
    # Try <title>
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
        # Remove common suffixes like "— SDK Documentation (1.4.0)"
        title = re.sub(r"\s*[—–|]\s*SDK Documentation.*$", "", title).strip()
        if title:
            return title

    # Try first <h1>
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True).rstrip("¶")

    return "Untitled"


def extract_links(soup, base_url):
    """Extract all internal documentation links from a page."""
    links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]

        # Skip anchors, javascript, mailto
        if href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue

        # Resolve relative URLs
        full_url = urljoin(base_url, href)

        # Remove fragment
        full_url, _ = urldefrag(full_url)

        # Strip trailing slash for consistency (keep root as-is)
        if full_url != START_URL:
            full_url = full_url.rstrip("/")

        # Only keep same-domain URLs
        parsed = urlparse(full_url)
        if parsed.netloc != DOMAIN:
            continue

        # Skip non-HTML resources
        _, ext = os.path.splitext(parsed.path)
        if ext.lower() in SKIP_EXTENSIONS:
            continue

        links.add(full_url)

    return links


# ─── Crawl Logic ──────────────────────────────────────────────────────────────

def crawl(max_depth, max_pages, resume=False):
    """BFS crawl of the Cerebras SDK documentation."""
    session = requests.Session()
    session.headers.update(HEADERS)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # If resuming, collect already-downloaded pages
    existing_files = set()
    if resume:
        for root, _, files in os.walk(OUTPUT_DIR):
            for f in files:
                if f.endswith(".md") and f != "index.md":
                    rel = os.path.relpath(os.path.join(root, f), OUTPUT_DIR)
                    existing_files.add(rel)
        if existing_files:
            print(f"Resume mode: {len(existing_files)} pages already downloaded, will skip them.")

    # BFS queue: (url, depth)
    queue = deque([(START_URL, 0)])
    visited = set()
    pages = []

    print(f"Crawl target : {START_URL}")
    print(f"Output dir   : {os.path.abspath(OUTPUT_DIR)}")
    print(f"Max depth    : {max_depth}")
    print(f"Max pages    : {max_pages}")
    print("-" * 70)

    while queue and len(pages) < max_pages:
        url, depth = queue.popleft()

        # Normalize
        url_norm = url.rstrip("/") if url != START_URL else url
        if url_norm in visited:
            continue
        visited.add(url_norm)

        if depth > max_depth:
            continue

        # Check resume
        filepath = url_to_filepath(url)
        if resume and filepath in existing_files:
            # Still need to extract links for further crawling
            pages.append({"url": url, "filepath": filepath, "title": "(cached)", "depth": depth})
            # We don't re-fetch, but we also can't extract links from cached files
            # So we skip link extraction for cached pages
            continue

        # Fetch
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  FAIL: {url} ({e})")
            continue

        content_type = resp.headers.get("content-type", "")
        if "text/html" not in content_type:
            continue

        html = resp.text
        soup = BeautifulSoup(html, "html.parser")

        title = get_page_title(soup)
        markdown = html_to_markdown(html, url)

        # Strip leading duplicate h1 headings (we add our own in the header)
        escaped_title = re.escape(title)
        markdown = re.sub(
            rf"^(# {escaped_title}\s*\n\s*)+",
            "",
            markdown,
        ).lstrip()

        if not markdown or len(markdown) < 30:
            print(f"  SKIP (empty): {url}")
            continue

        # Save the markdown file
        full_path = os.path.join(OUTPUT_DIR, filepath)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        crawled_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = (
            f"---\n"
            f"title: \"{title}\"\n"
            f"url: \"{url}\"\n"
            f"crawled_at: \"{crawled_at}\"\n"
            f"---\n\n"
            f"# {title}\n\n"
            f"> Source: [{url}]({url})\n\n"
        )
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(header + markdown + "\n")

        pages.append({"url": url, "filepath": filepath, "title": title, "depth": depth})
        print(f"  [{len(pages):3d}] depth={depth}  {filepath}")

        # Extract and enqueue child links
        if depth < max_depth:
            child_links = extract_links(soup, url)
            for link in child_links:
                link_norm = link.rstrip("/") if link != START_URL else link
                if link_norm not in visited:
                    queue.append((link, depth + 1))

        # Rate-limit
        time.sleep(REQUEST_DELAY)

    return pages


# ─── Index Generation ─────────────────────────────────────────────────────────

# Logical ordering and display names for top-level sections
SECTION_ORDER = [
    ("_root", "Getting Started"),
    ("installation-guide", "Installation"),
    ("computing-with-cerebras", "Conceptual View"),
    ("tensor-streaming", "Tensor Streaming"),
    ("appliance-mode", "Appliance Mode"),
    ("csl", "CSL (Cerebras Streaming Language)"),
    ("debug", "Debugging"),
    ("api-docs", "API Reference"),
    ("sdk-release-notes", "Release Notes"),
]

# Sub-section ordering within CSL
CSL_SUBSECTION_ORDER = [
    ("language_index", "Language Guide - Overview"),
    ("language", "Language Reference"),
    ("tutorials", "Tutorials"),
    ("code-examples", "Code Examples"),
    ("csl-compiler", "Compiler"),
    ("working-with-code-samples", "Working with Code Samples"),
]


def generate_index(pages):
    """Generate index.md with a structured table of contents."""
    index_path = os.path.join(OUTPUT_DIR, "index.md")

    # Group pages by top-level directory
    groups = {}
    for page in pages:
        fp = page["filepath"]
        parts = fp.replace("\\", "/").split("/")
        if len(parts) > 1:
            group = parts[0]
        else:
            group = "_root"
        groups.setdefault(group, []).append(page)

    with open(index_path, "w", encoding="utf-8") as f:
        f.write("# Cerebras SDK Documentation\n\n")
        f.write(f"> Crawled from: <{START_URL}>\n>\n")
        f.write(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n>\n")
        f.write(f"> Total pages: {len(pages)}\n\n")
        f.write("---\n\n")

        # Write sections in defined order
        written_groups = set()
        for group_key, display_name in SECTION_ORDER:
            if group_key not in groups:
                continue
            written_groups.add(group_key)

            f.write(f"## {display_name}\n\n")
            section_pages = groups[group_key]

            if group_key == "csl":
                # CSL has rich sub-structure; organize it
                _write_csl_section(f, section_pages)
            else:
                for page in sorted(section_pages, key=lambda p: p["filepath"]):
                    rel_path = page["filepath"]
                    f.write(f"- [{page['title']}]({rel_path})\n")
            f.write("\n")

        # Any remaining groups not in SECTION_ORDER
        for group_key in sorted(groups.keys()):
            if group_key in written_groups:
                continue
            display_name = group_key.replace("-", " ").replace("_", " ").title()
            f.write(f"## {display_name}\n\n")
            for page in sorted(groups[group_key], key=lambda p: p["filepath"]):
                rel_path = page["filepath"]
                f.write(f"- [{page['title']}]({rel_path})\n")
            f.write("\n")

    print(f"\nIndex written: {os.path.abspath(index_path)}")


def _write_csl_section(f, pages):
    """Write the CSL section with sub-grouping."""
    # Sub-group by second path component
    sub_groups = {}
    for page in pages:
        parts = page["filepath"].replace("\\", "/").split("/")
        if len(parts) > 2:
            sub_key = parts[1]
        else:
            sub_key = "_csl_root"
        sub_groups.setdefault(sub_key, []).append(page)

    written = set()
    for sub_key, display_name in CSL_SUBSECTION_ORDER:
        matching = sub_groups.get(sub_key, [])
        if not matching:
            # Check if it's a single file like csl/language_index.md
            matching = [p for p in sub_groups.get("_csl_root", [])
                        if sub_key in p["filepath"]]
        if not matching:
            continue
        written.add(sub_key)

        f.write(f"### {display_name}\n\n")
        for page in sorted(matching, key=lambda p: p["filepath"]):
            rel_path = page["filepath"]
            f.write(f"- [{page['title']}]({rel_path})\n")
        f.write("\n")

    # Remaining CSL sub-groups
    for sub_key in sorted(sub_groups.keys()):
        if sub_key in written or sub_key == "_csl_root":
            continue
        remaining = sub_groups[sub_key]
        display_name = sub_key.replace("-", " ").replace("_", " ").title()
        f.write(f"### {display_name}\n\n")
        for page in sorted(remaining, key=lambda p: p["filepath"]):
            rel_path = page["filepath"]
            f.write(f"- [{page['title']}]({rel_path})\n")
        f.write("\n")

    # Any root-level CSL pages not yet written
    csl_root = sub_groups.get("_csl_root", [])
    unwritten = [p for p in csl_root
                 if not any(sk in p["filepath"] for sk, _ in CSL_SUBSECTION_ORDER)]
    if unwritten:
        for page in sorted(unwritten, key=lambda p: p["filepath"]):
            rel_path = page["filepath"]
            f.write(f"- [{page['title']}]({rel_path})\n")
        f.write("\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Crawl Cerebras SDK docs to Markdown")
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH,
                        help=f"Maximum crawl depth (default: {DEFAULT_MAX_DEPTH})")
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES,
                        help=f"Maximum pages to crawl (default: {DEFAULT_MAX_PAGES})")
    parser.add_argument("--resume", action="store_true",
                        help="Skip pages that already exist on disk")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: Cerebras_Docs)")
    args = parser.parse_args()

    if args.output_dir:
        OUTPUT_DIR = args.output_dir

    pages = crawl(
        max_depth=args.max_depth,
        max_pages=args.max_pages,
        resume=args.resume,
    )

    if pages:
        generate_index(pages)
        print(f"\nDone! {len(pages)} pages saved to: {os.path.abspath(OUTPUT_DIR)}")
        print(f"Open index.md for the table of contents.")
    else:
        print("\nNo pages were crawled.")


if __name__ == "__main__":
    main()
