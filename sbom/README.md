# SBOM for livekit/agents

Generated with [CycloneDX Python](https://github.com/CycloneDX/cyclonedx-python) from the `uv` lockfile.

## Files

- `requirements.txt` — Resolved dependencies exported from `uv.lock` (`uv export --format requirements-txt`).
- `sbom.cdx.json` — CycloneDX 1.6 SBOM (JSON).
- `sbom.cdx.xml` — CycloneDX 1.6 SBOM (XML).

Both SBOMs contain 251 components across all workspace extras (all plugins resolved).

## Regenerating

```bash
uv export --format requirements-txt --no-hashes --no-emit-project -o sbom/requirements.txt
uv tool run --from cyclonedx-bom cyclonedx-py requirements \
    sbom/requirements.txt \
    --pyproject livekit-agents/pyproject.toml \
    --mc-type library \
    --of JSON -o sbom/sbom.cdx.json
```
