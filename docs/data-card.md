# Data card: governed public-source collection

This project uses the crawler only for individually listed, public research sources. It is not a general web crawler and must not be pointed at user, account, authenticated, or business-content pages.

## Access and collection boundaries

- Each manifest URL must be HTTPS and its hostname must exactly match the manifest allowlist. Hostnames must use strict canonical DNS-label grammar; `localhost`, `.localhost`, `.local`, every IPv4/IPv6 literal, and inet_aton-compatible decimal/octal/hexadecimal spellings such as `127.1`, `0177.0.0.1`, and `0x7f000001` are rejected even if listed. This prevents literal private/non-global address access, but DNS rebinding remains a deployment concern: run the crawler with network egress controls that allow only approved public destinations. Subdomains, credentials, fragments, non-standard ports, and account-like paths are also rejected. The route-segment blocklist covers account, auth, dashboard, login, profile, settings, and user surfaces after full percent-decoding, Unicode normalization, and case normalization. Credential-bearing query parameter names (for example `access_token`, `session_id`, and `api-key`) are rejected without logging their values; ordinary public parameters such as `q` and `page` remain allowed.
- The crawler sends one identified user agent, fetches and applies `robots.txt`, never authenticates, never follows redirects, and limits robots files and documents to bounded response sizes.
- A source manifest is fully validated before the first request. URLs are fetched one at a time; there is no discovery, link traversal, pagination, or broad crawling.
- Only the original response body is written as a deterministic `.raw` file. Output and metadata directories plus final files cannot be symlinks; writes are staged and atomically replaced. A transaction marker is published only after both the raw file and metadata sidecar have committed. Downstream code must treat a raw artifact as valid only when `is_complete_artifact(...)` verifies that marker and both hashes; unmarked raw files are incomplete and must not be consumed. The crawler does not parse, label, normalize, or convert downloaded material into training data.

## Required provenance record

For each raw file, the crawler stores a JSON metadata sidecar under `data/raw/metadata/` (or the selected output directory) containing:

- source URL and retrieval timestamp;
- SHA-256 hash of the raw content;
- declared source type;
- license or rights note; and
- terms-review status.

The metadata sidecar is provenance, not training content. Raw downloads remain quarantined from SFT/RL data until an authorized owner has reviewed the source terms, de-identified the material, documented the transformation, and approved the resulting processed dataset. Business data, user content, labels, policy decisions, and oracle fields are never accepted by this crawler.
