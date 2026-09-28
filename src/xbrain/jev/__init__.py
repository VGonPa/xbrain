"""Jev (TypeSafe AI) topic assessment.

A side-car layer: one typed-question call per item asks which vocabulary topics the post
belongs to (one Noul per topic, one Choice for the primary) and stores the probabilities in
`data/jev/` — next to, never inside, `items.json`. See ARCHITECTURE.md § jev.
"""
