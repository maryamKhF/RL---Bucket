"""Shared names and version for routing recovery diagnostics."""

RECOVERY_VERSION = "local-prefix-v2"
RECOVERY_COUNT_FIELDS = (
    "partial_backtrack_retry_count",
    "source_retry_count",
    "preserved_prefix_hops",
    "backtracked_hops",
)


def recovery_details(info):
    """Return serializable transaction diagnostics, including empty trials."""
    return {
        "recovery_version": RECOVERY_VERSION,
        **{key: int(info.get(key, 0) or 0) for key in RECOVERY_COUNT_FIELDS},
        "recovery_events": info.get("recovery_events", []),
    }
