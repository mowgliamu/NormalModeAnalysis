"""Shared pytest fixtures."""

from __future__ import annotations

import jax

# Force float64 for tests too
jax.config.update("jax_enable_x64", True)
