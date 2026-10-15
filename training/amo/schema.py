"""Schema id matching across the AOM -> AMO rename.

Ids were spelled ``aom.*`` before the project abbreviation became AMO. Datasets,
checkpoints and models written then must keep loading, so a legacy ``aom.`` prefix
reads as the same ``amo.`` id. Writers always emit ``amo.``.

rename-compat: whole file
"""

_LEGACY, _CURRENT = "aom.", "amo."  # rename-compat


def canonical_id(value):
    """Return ``value`` with a legacy prefix rewritten; non-strings pass through."""
    if isinstance(value, str) and value.startswith(_LEGACY):
        return _CURRENT + value[len(_LEGACY):]
    return value


def schema_is(value, expected):
    return canonical_id(value) == expected


def same_id(a, b):
    return canonical_id(a) == canonical_id(b)
