"""
NVD (National Vulnerability Database) CVE Lookup Tool.
Queries the official NIST NVD API v2.0 to retrieve CVE details.
This is a free, authoritative source that doesn't require Google API keys.
"""
import requests
import logging
from typing import Optional
from pydantic import BaseModel
from langchain_core.tools import Tool

logger = logging.getLogger(__name__)

NVD_API_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"


class NVDLookupArgs(BaseModel):
    cve_id: str


def nvd_cve_lookup_func(cve_id: str) -> str:
    """
    Query the NVD API for a specific CVE ID and return structured information.
    
    Args:
        cve_id: The CVE identifier (e.g., "CVE-2021-44228")
    Returns:
        A formatted string with CVE details from NVD.
    """
    cve_id = cve_id.strip().upper()
    
    # Validate CVE ID format
    if not cve_id.startswith("CVE-"):
        return f"Invalid CVE ID format: '{cve_id}'. Expected format: CVE-YYYY-NNNNN"

    try:
        response = requests.get(
            NVD_API_BASE,
            params={"cveId": cve_id},
            timeout=15,
            headers={"User-Agent": "CybersleuthForensicAgent/1.0"}
        )

        if response.status_code == 404:
            return f"CVE {cve_id} not found in NVD database."
        
        if response.status_code == 403:
            return f"NVD API rate limit reached. Try again in 30 seconds."

        if response.status_code != 200:
            return f"NVD API returned status {response.status_code} for {cve_id}."

        data = response.json()
        vulnerabilities = data.get("vulnerabilities", [])
        
        if not vulnerabilities:
            return f"No vulnerability data found for {cve_id}."

        cve_data = vulnerabilities[0].get("cve", {})
        
        # Extract description
        descriptions = cve_data.get("descriptions", [])
        description = next(
            (d["value"] for d in descriptions if d.get("lang") == "en"),
            descriptions[0]["value"] if descriptions else "No description available"
        )

        # Extract CVSS score
        metrics = cve_data.get("metrics", {})
        cvss_info = "N/A"
        
        # Try CVSS v3.1 first, then v3.0, then v2.0
        for version_key in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
            if version_key in metrics and metrics[version_key]:
                cvss_data = metrics[version_key][0].get("cvssData", {})
                score = cvss_data.get("baseScore", "N/A")
                severity = cvss_data.get("baseSeverity", "N/A")
                vector = cvss_data.get("vectorString", "")
                cvss_info = f"{score} ({severity}) - {vector}"
                break

        # Extract weaknesses (CWE)
        weaknesses = cve_data.get("weaknesses", [])
        cwe_list = []
        for w in weaknesses:
            for desc in w.get("description", []):
                if desc.get("value", "").startswith("CWE-"):
                    cwe_list.append(desc["value"])
        cwe_info = ", ".join(cwe_list) if cwe_list else "N/A"

        # Extract affected configurations
        configs = cve_data.get("configurations", [])
        affected_products = []
        for config in configs:
            for node in config.get("nodes", []):
                for cpe_match in node.get("cpeMatch", []):
                    criteria = cpe_match.get("criteria", "")
                    if cpe_match.get("vulnerable", False):
                        # Parse CPE string: cpe:2.3:a:vendor:product:version:...
                        parts = criteria.split(":")
                        if len(parts) >= 6:
                            vendor = parts[3]
                            product = parts[4]
                            version = parts[5] if parts[5] != "*" else "all versions"
                            affected_products.append(f"{vendor}/{product} ({version})")
        
        products_info = "; ".join(affected_products[:5]) if affected_products else "See NVD for details"

        # Format result
        result = f"""=== NVD CVE Lookup: {cve_id} ===
Description: {description}
CVSS Score: {cvss_info}
Weaknesses: {cwe_info}
Affected Products: {products_info}
Published: {cve_data.get('published', 'N/A')}
Last Modified: {cve_data.get('lastModified', 'N/A')}
Reference: https://nvd.nist.gov/vuln/detail/{cve_id}
"""
        return result

    except requests.Timeout:
        return f"NVD API request timed out for {cve_id}. Try again later."
    except requests.ConnectionError:
        return f"Could not connect to NVD API. Check internet connection."
    except Exception as e:
        logger.error(f"NVD lookup error for {cve_id}: {e}")
        return f"Error looking up {cve_id} in NVD: {str(e)}"


nvd_cve_lookup = Tool(
    name="nvd_cve_lookup",
    description="""Look up a specific CVE in the National Vulnerability Database (NVD).
    Use this tool when you have identified a potential CVE ID and want to verify it 
    against the official database. Returns description, CVSS score, affected products, 
    and weakness types. This is more reliable and cheaper than web search for CVE verification.
    Args:
        cve_id: The CVE identifier to look up (e.g., "CVE-2021-44228")
    """,
    args_schema=NVDLookupArgs,
    func=nvd_cve_lookup_func,
)

__all__ = ["nvd_cve_lookup", "nvd_cve_lookup_func"]
