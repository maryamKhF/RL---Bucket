"""Deterministic, resumable sampling of costly policy sensitivity calculations."""

import hashlib
import json
import math


DEFAULT_SAMPLING = {
    "version": "routing-all-policy-sampled-v1",
    "success_probability": 0.05,
    "failure_probability": 0.25,
    "partial_backtrack_probability": 0.25,
}


def validate_sampling(settings):
    if not isinstance(settings, dict) or settings.get("version") != DEFAULT_SAMPLING["version"]:
        raise ValueError("Unsupported round_one_xai sampling configuration.")
    for group in ("success", "failure", "partial_backtrack"):
        value = settings.get(group + "_probability")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("XAI sampling probabilities must be numbers in [0, 1].")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("XAI sampling probabilities must be numbers in [0, 1].")


def policy_sample(cfg, identity, routing):
    """Hash within outcome/recovery strata without advancing any experiment RNG.

    No settings means the original exhaustive protocol, including on old resume.
    Selection uses actual outcomes, so sampled aggregate means need weighting.
    """
    decision = routing.get("decision", {})
    if (decision.get("partial_backtrack_retry_count", 0) or
            decision.get("partial_backtrack_count", 0)):
        stratum = "partial_backtrack"
    else:
        stratum = "success" if decision.get("success") else "failure"
    settings = cfg.get("round_one_xai")
    if settings is None:
        probability, version = 1.0, "exhaustive-v1"
    else:
        validate_sampling(settings)
        probability, version = float(settings[stratum + "_probability"]), settings["version"]
    encoded = json.dumps([version, int(cfg.get("seed", 0)), identity, stratum],
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    score = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
    selected = probability == 1 or score < probability * 2**64
    return {"method": "routing_audit_only", "sampled": bool(selected),
            "sampling": {"version": version, "stratum": stratum,
                         "probability": probability,
                         "selection": "deterministic hash within phase/rate/outcome/recovery strata"}}
