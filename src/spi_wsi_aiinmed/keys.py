#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive key handling for notebook-first workflows.

The package intentionally does not store API keys in files. The recommended
notebook option is to ask the user at runtime using getpass, then keep the keys
only in the current Python process environment.
"""
from __future__ import annotations

import os
from getpass import getpass
from typing import Dict, Optional


def ask_user_for_keys(
    *,
    ask_anthropic: bool = True,
    ask_hf: bool = True,
    ask_entrez_email: bool = True,
    default_entrez_email: str = "",
    overwrite: bool = False,
) -> Dict[str, str]:
    """Ask for API keys interactively and store them in os.environ.

    This is the only key-entry option exposed in example notebooks.
    Values are not written to disk.
    """
    out: Dict[str, str] = {}

    if ask_anthropic and (overwrite or not os.environ.get("ANTHROPIC_API_KEY")):
        val = getpass("Enter Anthropic API key (input hidden): ").strip()
        if val:
            os.environ["ANTHROPIC_API_KEY"] = val
    out["ANTHROPIC_API_KEY"] = "SET" if os.environ.get("ANTHROPIC_API_KEY") else "MISSING"

    if ask_hf and (overwrite or not os.environ.get("HF_TOKEN")):
        val = getpass("Enter Hugging Face token for CONCH, if required (blank to skip): ").strip()
        if val:
            os.environ["HF_TOKEN"] = val
    out["HF_TOKEN"] = "SET" if os.environ.get("HF_TOKEN") else "MISSING_OR_NOT_REQUIRED"

    if ask_entrez_email and (overwrite or not os.environ.get("ENTREZ_EMAIL")):
        prompt = "Enter Entrez/PubMed email"
        if default_entrez_email:
            prompt += f" [{default_entrez_email}]"
        prompt += ": "
        val = input(prompt).strip() or default_entrez_email
        if val:
            os.environ["ENTREZ_EMAIL"] = val
    out["ENTREZ_EMAIL"] = os.environ.get("ENTREZ_EMAIL", "")
    return out


def require_key(name: str, value: Optional[str] = None, allow_missing: bool = False) -> str:
    """Return a supplied or environment key, raising a helpful error if absent."""
    resolved = value or os.environ.get(name, "")
    if not resolved and not allow_missing:
        raise RuntimeError(
            f"Missing {name}. In a notebook, run ask_user_for_keys(); "
            f"in a terminal, set the {name} environment variable."
        )
    return resolved
