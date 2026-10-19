"""Build the real-figure corpus: CC-BY open-access articles whose figure caption says "forest plot".

Search: Europe PMC REST (no auth). Figure files + JATS XML: the PMC Open Access bucket on S3
(pmc-oa-opendata, public, no auth). Europe PMC's own image endpoint 403s scripted clients.
Writes data/corpus/<pmcid>_<figid>.<ext> (gitignored) and data/manifest.json (committed).
"""
import json, re, sys, time, pathlib
import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "corpus"
OUT.mkdir(parents=True, exist_ok=True)
S3 = "https://pmc-oa-opendata.s3.amazonaws.com"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
QUERY = '"forest plot" AND LICENSE:"cc by" AND OPEN_ACCESS:y AND ("hazard ratio" OR "odds ratio" OR "risk ratio" OR "relative risk")'
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 150
FIG_RE = re.compile(r"<fig\b.*?</fig>", re.S)


def pmcids():
    cursor = "*"
    while True:
        r = requests.get(EPMC, params={"query": QUERY, "format": "json", "pageSize": 100,
                                       "resultType": "lite", "cursorMark": cursor}, timeout=60)
        r.raise_for_status()
        j = r.json()
        for x in j["resultList"]["result"]:
            if x.get("pmcid"):
                yield x["pmcid"], x.get("doi"), x.get("title")
        nxt = j.get("nextCursorMark")
        if not nxt or nxt == cursor:
            return
        cursor = nxt


def article_xml(pmcid):
    for v in (1, 2, 3):
        r = requests.get(f"{S3}/{pmcid}.{v}/{pmcid}.{v}.xml", timeout=60)
        if r.status_code == 200:
            return v, r.text
    return None, None


def main():
    manifest, seen_articles = [], 0
    for pmcid, doi, title in pmcids():
        if len(manifest) >= TARGET:
            break
        seen_articles += 1
        v, xml = article_xml(pmcid)
        if not xml:
            continue
        n_from_article = 0
        for fig in FIG_RE.findall(xml):
            cap = re.sub(r"<[^>]+>", " ", fig)
            if not re.search(r"forest plot", cap, re.I):
                continue
            m = re.search(r'xlink:href="([^"]+)"', fig)
            if not m or n_from_article >= 2:  # at most 2 figures/article: avoid one meta-analysis dominating
                continue
            href = m.group(1)
            name = href if re.search(r"\.(jpe?g|png|webp|gif)$", href, re.I) else None
            if name is None:
                continue
            dest = OUT / f"{pmcid}_{pathlib.Path(name).name}"
            if not dest.exists():
                r = requests.get(f"{S3}/{pmcid}.{v}/{name}", timeout=120)
                if r.status_code != 200:
                    continue
                dest.write_bytes(r.content)
            manifest.append({"id": dest.stem, "file": dest.name, "pmcid": pmcid, "doi": doi,
                             "article_title": title, "license": "CC BY",
                             "caption": re.sub(r"\s+", " ", cap).strip()[:300]})
            n_from_article += 1
        print(f"[{len(manifest)}/{TARGET}] {pmcid} articles_seen={seen_articles}", flush=True)
        time.sleep(0.1)
    (ROOT / "data" / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("wrote", len(manifest), "figures")


if __name__ == "__main__":
    main()
