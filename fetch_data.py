"""
Step 1: download the data (run once on your own computer).

- knowledge/*.txt : astronomy articles from Wikipedia (for the Q&A part)
- data/kepler.csv : NASA Kepler candidates table (for the classifier)

Safe to re-run: files that were already downloaded are skipped, and
temporary network errors are retried automatically.

Run:  python fetch_data.py
"""
import os
import re
import time

import requests

HEADERS = {"User-Agent": "astro-assistant/1.0 (student project; contact: azin)"}

TOPICS = [
    "Exoplanet", "Kepler space telescope", "Transit (astronomy)",
    "Methods of detecting exoplanets", "Habitable zone", "Red dwarf",
    "Super-Earth", "Hot Jupiter", "TRAPPIST-1", "Proxima Centauri b",
    "TESS (satellite)", "James Webb Space Telescope", "Kepler-452b",
    "Radial velocity", "Stellar classification", "Main sequence",
    "Gas giant", "Terrestrial planet", "Circumstellar habitable zone",
    "Planetary equilibrium temperature", "Light curve", "Binary star",
]


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def get_with_retries(url, params, tries=4, timeout=60):
    """GET with retries; returns the response or None if it keeps failing."""
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if r.status_code == 200 and r.text.strip():
                return r
            print(f"    attempt {attempt}: status {r.status_code}, retrying...")
        except requests.RequestException as e:
            print(f"    attempt {attempt}: {type(e).__name__}, retrying...")
        time.sleep(3 * attempt)
    return None


def fetch_wikipedia():
    os.makedirs("knowledge", exist_ok=True)
    saved, failed = 0, []
    for title in TOPICS:
        path = f"knowledge/{slug(title)}.txt"
        if os.path.exists(path):
            print(f"  already have: {title}")
            saved += 1
            continue

        r = get_with_retries(
            "https://en.wikipedia.org/w/api.php",
            {
                "action": "query", "prop": "extracts", "explaintext": 1,
                "titles": title, "format": "json", "redirects": 1,
            },
        )
        try:
            pages = r.json()["query"]["pages"]
            text = next(iter(pages.values())).get("extract", "")
        except Exception:
            text = ""

        if len(text) < 200:
            print(f"  FAILED: {title}")
            failed.append(title)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"{title}\n\n{text}")
            print(f"  saved: {title} ({len(text)} chars)")
            saved += 1
        time.sleep(1.5)  # be gentle with Wikipedia

    print(f"Articles ready: {saved}/{len(TOPICS)}")
    if failed:
        print("Failed (run the script again to retry):", ", ".join(failed))


def fetch_kepler():
    os.makedirs("data", exist_ok=True)
    path = "data/kepler.csv"
    if os.path.exists(path):
        print("  already have: data/kepler.csv")
        return
    query = (
        "select koi_disposition,koi_period,koi_duration,koi_depth,koi_prad,"
        "koi_teq,koi_insol,koi_model_snr,koi_steff,koi_slogg,koi_srad "
        "from cumulative "
        "where koi_disposition in ('CONFIRMED','FALSE POSITIVE')"
    )
    r = get_with_retries(
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
        {"query": query, "format": "csv"},
        tries=3, timeout=120,
    )
    if r is None:
        print("  FAILED: could not download the Kepler table from NASA.")
        print("  Run the script again later, or tell me and we will use another source.")
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(r.text)
    print("  saved: data/kepler.csv")


if __name__ == "__main__":
    print("Downloading Wikipedia articles...")
    fetch_wikipedia()
    print("Downloading Kepler table from NASA...")
    fetch_kepler()
    print("Done.")
