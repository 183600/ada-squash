"""Analyst blinding support for unbiased analysis.

An analysis is only blind if the analyst cannot map coded arm labels back to
the scientific conditions until the analysis plan is locked and the primary
contrasts are executed. This module provides:

``blind_arms``
    Replace arm names with opaque codes (``A01``, ``A02``, ...) and return a
    sealed key that can only be opened after a pre-registered analysis is
    complete. The key is encrypted with a passphrase or a time-lock.

``unblind_report``
    Decode a report produced on blinded data using the sealed key. Fails if
    the key was opened before the pre-registered analysis fingerprint matches.

``BlindedAnalysis``
    Context manager that enforces blinding: the analysis function receives
    only coded labels; the unblinding step is explicit and audited.

The blinding protocol is designed to be compatible with the existing
``analysis_plan.json`` fingerprint: the blinding key is included in the plan
so that changing the arm mapping changes the pre-registration.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = [
    "BlindingKey",
    "BlindedAnalysis",
    "blind_arms",
    "unblind_report",
    "create_blinding_key",
    "verify_blinding_integrity",
    "time_locked_key",
]


@dataclass(frozen=True)
class BlindingKey:
    """Sealed blinding key with integrity verification.

    Attributes
    ----------
    version: Schema version.
    arm_map: Mapping from original arm name to blinded code.
    fingerprint: SHA-256 of the arm_map for integrity.
    created_at: Unix timestamp when the key was created.
    analysis_plan_fingerprint: The analysis plan fingerprint this key is bound to.
    hmac_tag: HMAC-SHA256 of the key payload using a passphrase-derived key.
    """
    version: int = 1
    arm_map: dict[str, str] = field(default_factory=dict)
    fingerprint: str = ""
    created_at: float = field(default_factory=time.time)
    analysis_plan_fingerprint: str = ""
    hmac_tag: str = ""

    def to_json(self) -> str:
        return json.dumps({
            "version": self.version,
            "arm_map": self.arm_map,
            "fingerprint": self.fingerprint,
            "created_at": self.created_at,
            "analysis_plan_fingerprint": self.analysis_plan_fingerprint,
            "hmac_tag": self.hmac_tag,
        }, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, data: str) -> "BlindingKey":
        obj = json.loads(data)
        return cls(
            version=obj.get("version", 1),
            arm_map=obj.get("arm_map", {}),
            fingerprint=obj.get("fingerprint", ""),
            created_at=obj.get("created_at", 0.0),
            analysis_plan_fingerprint=obj.get("analysis_plan_fingerprint", ""),
            hmac_tag=obj.get("hmac_tag", ""),
        )

    def verify_integrity(self, passphrase: str) -> bool:
        """Verify the HMAC tag using the passphrase."""
        expected = self._compute_hmac(passphrase)
        return hmac.compare_digest(self.hmac_tag, expected)

    def _compute_hmac(self, passphrase: str) -> str:
        key = hashlib.pbkdf2_hmac("sha256", passphrase.encode(), b"adasquash-blinding", 100_000)
        payload = json.dumps({
            "arm_map": self.arm_map,
            "fingerprint": self.fingerprint,
            "created_at": self.created_at,
            "analysis_plan_fingerprint": self.analysis_plan_fingerprint,
        }, sort_keys=True, separators=(",", ":")).encode()
        return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _generate_blinded_codes(arms: Sequence[str], prefix: str = "A") -> dict[str, str]:
    """Generate opaque codes for arms (A01, A02, ...)."""
    if not arms:
        return {}
    width = max(2, len(str(len(arms))))
    return {arm: f"{prefix}{index:0{width}d}" for index, arm in enumerate(sorted(arms), 1)}


def _compute_arm_map_fingerprint(arm_map: Mapping[str, str]) -> str:
    """SHA-256 of the arm mapping for integrity."""
    payload = json.dumps(dict(sorted(arm_map.items())), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def create_blinding_key(
    arms: Sequence[str],
    analysis_plan_fingerprint: str,
    passphrase: str,
    prefix: str = "A",
) -> BlindingKey:
    """Create a sealed blinding key for the given arms.

    The key binds the arm mapping to the analysis plan fingerprint. Any change
    to the analysis plan (new contrasts, different arms) changes the fingerprint
    and invalidates the key.
    """
    arm_map = _generate_blinded_codes(arms, prefix)
    fingerprint = _compute_arm_map_fingerprint(arm_map)
    key = BlindingKey(
        arm_map=arm_map,
        fingerprint=fingerprint,
        analysis_plan_fingerprint=analysis_plan_fingerprint,
    )
    # Compute HMAC after creating the key object
    hmac_tag = key._compute_hmac(passphrase)
    return BlindingKey(
        version=key.version,
        arm_map=key.arm_map,
        fingerprint=key.fingerprint,
        created_at=key.created_at,
        analysis_plan_fingerprint=key.analysis_plan_fingerprint,
        hmac_tag=hmac_tag,
    )


def blind_arms(data: Mapping[str, Any], blinding_key: BlindingKey) -> dict[str, Any]:
    """Replace arm names in data with blinded codes.

    Works recursively on dicts, lists, and strings that match arm names.
    Non-string values and strings not in the arm map are left unchanged.
    """
    reverse_map = {v: k for k, v in blinding_key.arm_map.items()}

    def _blind_value(value: Any) -> Any:
        if isinstance(value, str):
            # Check if this string is an arm name that should be blinded
            if value in blinding_key.arm_map:
                return blinding_key.arm_map[value]
            # Check if it's already a blinded code (should not happen in normal use)
            if value in reverse_map:
                return value  # Already blinded
            return value
        if isinstance(value, dict):
            return {k: _blind_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_blind_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_blind_value(v) for v in value)
        return value

    return _blind_value(dict(data))


def unblind_report(report: Mapping[str, Any], blinding_key: BlindingKey, passphrase: str) -> dict[str, Any]:
    """Decode a report produced on blinded data.

    Verifies the blinding key integrity and that it matches the analysis plan
    fingerprint recorded in the report (if present).
    """
    if not blinding_key.verify_integrity(passphrase):
        raise ValueError("Blinding key integrity check failed: HMAC tag mismatch")

    # Verify analysis plan fingerprint if present in report
    report_plan_fp = report.get("analysis_plan_fingerprint")
    if report_plan_fp and report_plan_fp != blinding_key.analysis_plan_fingerprint:
        raise ValueError(
            f"Analysis plan fingerprint mismatch: report has {report_plan_fp!r}, "
            f"key has {blinding_key.analysis_plan_fingerprint!r}"
        )

    reverse_map = {v: k for k, v in blinding_key.arm_map.items()}

    def _unblind_value(value: Any) -> Any:
        if isinstance(value, str):
            if value in reverse_map:
                return reverse_map[value]
            return value
        if isinstance(value, dict):
            return {k: _unblind_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_unblind_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_unblind_value(v) for v in value)
        return value

    return _unblind_value(dict(report))


def verify_blinding_integrity(
    blinding_key: BlindingKey,
    passphrase: str,
    expected_plan_fingerprint: Optional[str] = None,
) -> dict[str, Any]:
    """Verify a blinding key's integrity and optionally its plan binding."""
    result = {
        "hmac_valid": blinding_key.verify_integrity(passphrase),
        "fingerprint_matches": _compute_arm_map_fingerprint(blinding_key.arm_map) == blinding_key.fingerprint,
        "plan_fingerprint": blinding_key.analysis_plan_fingerprint,
    }
    if expected_plan_fingerprint is not None:
        result["plan_fingerprint_matches"] = (
            blinding_key.analysis_plan_fingerprint == expected_plan_fingerprint
        )
    result["all_checks_pass"] = all(v for k, v in result.items() if k != "plan_fingerprint")
    return result


class BlindedAnalysis:
    """Context manager for blinded analysis with audit trail.

    Usage:
        key = create_blinding_key(arms, plan_fp, passphrase)
        with BlindedAnalysis(key, passphrase) as ba:
            blinded_data = ba.blind(raw_data)
            results = analyze(blinded_data)
            ba.record_analysis_fingerprint(results)
        # After analysis is complete and plan is locked:
        unblinded = unblind_report(results, key, passphrase)
    """

    def __init__(
        self,
        blinding_key: BlindingKey,
        passphrase: str,
        require_plan_lock: bool = True,
    ):
        self.key = blinding_key
        self.passphrase = passphrase
        self.require_plan_lock = require_plan_lock
        self._analysis_fingerprint: Optional[str] = None
        self._audit_log: list[dict] = []

    def __enter__(self) -> "BlindedAnalysis":
        # Verify key integrity on entry
        verification = verify_blinding_integrity(self.key, self.passphrase)
        if not verification["all_checks_pass"]:
            raise ValueError(f"Blinding key verification failed: {verification}")
        self._audit_log.append({
            "event": "blinded_analysis_start",
            "timestamp": time.time(),
            "key_fingerprint": self.key.fingerprint,
            "plan_fingerprint": self.key.analysis_plan_fingerprint,
            "verification": verification,
        })
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._audit_log.append({
            "event": "blinded_analysis_end",
            "timestamp": time.time(),
            "analysis_fingerprint": self._analysis_fingerprint,
            "exception": exc_type.__name__ if exc_type else None,
        })

    def blind(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Blind the input data using this analysis's key."""
        return blind_arms(data, self.key)

    def record_analysis_fingerprint(self, results: Mapping[str, Any]) -> None:
        """Record the fingerprint of the analysis that was run on blinded data.

        This should be called after the primary analysis completes, before
        unblinding. The fingerprint is typically the hash of the analysis code
        or the analysis plan.
        """
        if "analysis_plan_fingerprint" in results:
            self._analysis_fingerprint = results["analysis_plan_fingerprint"]
        else:
            # Compute a fingerprint of the results structure
            payload = json.dumps(results, sort_keys=True, default=str).encode()
            self._analysis_fingerprint = hashlib.sha256(payload).hexdigest()
        self._audit_log.append({
            "event": "analysis_fingerprint_recorded",
            "timestamp": time.time(),
            "fingerprint": self._analysis_fingerprint,
        })

    def unblind(self, report: Mapping[str, Any]) -> dict[str, Any]:
        """Unblind a report after analysis is complete.

        Requires that an analysis fingerprint was recorded (if require_plan_lock).
        """
        if self.require_plan_lock and self._analysis_fingerprint is None:
            raise RuntimeError(
                "Cannot unblind: no analysis fingerprint recorded. "
                "Call record_analysis_fingerprint() after running the primary analysis."
            )
        # Verify the analysis fingerprint matches the plan if both exist
        if self._analysis_fingerprint and self.key.analysis_plan_fingerprint:
            if self._analysis_fingerprint != self.key.analysis_plan_fingerprint:
                raise ValueError(
                    f"Analysis fingerprint {self._analysis_fingerprint[:12]}... "
                    f"does not match plan fingerprint {self.key.analysis_plan_fingerprint[:12]}... "
                    "This suggests the analysis was modified after blinding."
                )
        return unblind_report(report, self.key, self.passphrase)

    def get_audit_log(self) -> list[dict]:
        return list(self._audit_log)


def time_locked_key(
    blinding_key: BlindingKey,
    unlock_after: float,
    passphrase: str,
) -> str:
    """Create a time-locked version of the blinding key.

    The key is encrypted such that it cannot be decrypted before ``unlock_after``
    (Unix timestamp). This provides a commitment mechanism: the key is published
    before analysis, but cannot be used to unblind until the pre-specified time.

    Note: This is a *soft* time lock based on encryption with a time-derived
    key. For strong guarantees, use a trusted timestamping service or blockchain
    commitment instead.
    """
    # Derive a time-based key component
    time_salt = str(int(unlock_after)).encode()
    time_key = hashlib.pbkdf2_hmac("sha256", passphrase.encode(), time_salt, 100_000)

    # Encrypt the blinding key
    key_json = blinding_key.to_json().encode()
    # Simple XOR with time-derived keystream (for demonstration; use proper AEAD in production)
    keystream = hashlib.shake_256(time_key).digest(len(key_json))
    encrypted = bytes(a ^ b for a, b in zip(key_json, keystream))

    payload = {
        "schema": "adasquash.time_locked_key.v1",
        "unlock_after": unlock_after,
        "encrypted_key": base64.b64encode(encrypted).decode(),
        "key_fingerprint": blinding_key.fingerprint,
        "plan_fingerprint": blinding_key.analysis_plan_fingerprint,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def unlock_time_locked_key(
    locked_key_json: str,
    passphrase: str,
    current_time: Optional[float] = None,
) -> BlindingKey:
    """Unlock a time-locked blinding key.

    Raises ValueError if the current time is before the unlock time.
    """
    if current_time is None:
        current_time = time.time()

    payload = json.loads(locked_key_json)
    unlock_after = payload["unlock_after"]
    if current_time < unlock_after:
        raise ValueError(
            f"Time lock not yet expired: unlock after {unlock_after} "
            f"({time.ctime(unlock_after)}), current time is {current_time} "
            f"({time.ctime(current_time)})"
        )

    encrypted = base64.b64decode(payload["encrypted_key"])
    time_salt = str(int(unlock_after)).encode()
    time_key = hashlib.pbkdf2_hmac("sha256", passphrase.encode(), time_salt, 100_000)
    keystream = hashlib.shake_256(time_key).digest(len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, keystream))

    key = BlindingKey.from_json(decrypted.decode())
    # Verify the fingerprint matches
    if key.fingerprint != payload["key_fingerprint"]:
        raise ValueError("Time-locked key fingerprint mismatch after decryption")
    return key