"""
VirusTotal IOC Lookup Tool.
Queries the VirusTotal API v3 to validate IPs, domains, and file hashes.
Requires VIRUSTOTAL_API_KEY set in the environment / .env file.
"""
import os
import logging
import requests
from pydantic import BaseModel
from langchain_core.tools import Tool

logger = logging.getLogger(__name__)

VT_API_BASE = "https://www.virustotal.com/api/v3"


class VTLookupArgs(BaseModel):
    ioc: str
    ioc_type: str  # "ip", "domain", or "hash"


def virustotal_ioc_lookup_func(ioc: str, ioc_type: str) -> str:
    """
    Query VirusTotal for a specific IOC (IP, domain, or file hash).

    Args:
        ioc:      The indicator value (e.g. "192.168.1.1", "evil.com", "abc123...")
        ioc_type: One of "ip", "domain", or "hash"
    Returns:
        Formatted string with VirusTotal verdict and key metadata.
    """
    api_key = os.getenv("VIRUSTOTAL_API_KEY", "").strip()
    if not api_key:
        return (
            "VirusTotal API key not configured. "
            "Set VIRUSTOTAL_API_KEY in your .env file to enable IOC validation."
        )

    ioc = ioc.strip()
    ioc_type = ioc_type.lower().strip()

    endpoint_map = {
        "ip":     f"/ip_addresses/{ioc}",
        "domain": f"/domains/{ioc}",
        "hash":   f"/files/{ioc}",
    }

    if ioc_type not in endpoint_map:
        return f"Invalid ioc_type '{ioc_type}'. Accepted values: ip, domain, hash."

    url = VT_API_BASE + endpoint_map[ioc_type]

    try:
        response = requests.get(
            url,
            headers={"x-apikey": api_key},
            timeout=15,
        )
    except requests.RequestException as e:
        return f"VirusTotal request failed: {str(e)}"

    if response.status_code == 401:
        return "VirusTotal API key is invalid or expired."
    if response.status_code == 404:
        return f"IOC '{ioc}' ({ioc_type}) not found in VirusTotal database."
    if response.status_code == 429:
        return "VirusTotal API rate limit reached. Try again later."
    if response.status_code != 200:
        return f"VirusTotal API returned HTTP {response.status_code} for '{ioc}'."

    attrs = response.json().get("data", {}).get("attributes", {})

    stats = attrs.get("last_analysis_stats", {})
    malicious  = stats.get("malicious", 0)
    suspicious = stats.get("suspicious", 0)
    total      = sum(stats.values()) if stats else 0

    if malicious > 0:
        verdict = "MALICIOUS"
    elif suspicious > 0:
        verdict = "SUSPICIOUS"
    else:
        verdict = "CLEAN"

    lines = [
        f"VirusTotal report for {ioc_type.upper()} '{ioc}':",
        f"  Verdict    : {verdict}",
        f"  Detections : {malicious} malicious, {suspicious} suspicious / {total} engines",
    ]

    if ioc_type == "ip":
        lines.append(f"  Country    : {attrs.get('country', 'N/A')}")
        lines.append(f"  ASN        : {attrs.get('asn', 'N/A')} ({attrs.get('as_owner', 'N/A')})")

    elif ioc_type == "domain":
        lines.append(f"  Registrar  : {attrs.get('registrar', 'N/A')}")
        cats = attrs.get("categories", {})
        if cats:
            lines.append(f"  Categories : {', '.join(set(cats.values()))}")

    elif ioc_type == "hash":
        lines.append(f"  File type  : {attrs.get('type_description', 'N/A')}")
        lines.append(f"  File size  : {attrs.get('size', 'N/A')} bytes")
        threat_cls = attrs.get("popular_threat_classification", {})
        if threat_cls:
            lines.append(f"  Threat label: {threat_cls.get('suggested_threat_label', 'N/A')}")

    # Top malicious engine detections (up to 5)
    engine_results = attrs.get("last_analysis_results", {})
    top_hits = [
        f"{engine}: {info.get('result', 'N/A')}"
        for engine, info in engine_results.items()
        if info.get("category") == "malicious"
    ][:5]
    if top_hits:
        lines.append(f"  Top hits   : {'; '.join(top_hits)}")

    return "\n".join(lines)


virustotal_ioc_lookup = Tool(
    name="virustotal_ioc_lookup",
    description="""Validate a suspicious IP address, domain, or file hash against VirusTotal threat intelligence.
    Use this tool to confirm whether an observed network IOC is known to be malicious before including it in the report.
    Args:
        ioc:      The indicator value to look up (IP address, domain name, or MD5/SHA1/SHA256 hash).
        ioc_type: The type of indicator — must be one of: "ip", "domain", "hash".
    Returns:
        VirusTotal verdict (MALICIOUS / SUSPICIOUS / CLEAN) with detection counts and metadata.
    """,
    args_schema=VTLookupArgs,
    func=virustotal_ioc_lookup_func,
)

__all__ = ["virustotal_ioc_lookup", "virustotal_ioc_lookup_func"]
