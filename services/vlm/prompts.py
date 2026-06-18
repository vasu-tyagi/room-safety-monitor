"""VLM prompt templates (Slice 5).

SAFETY_ANALYSIS is the main prompt for L3 analysis. {kb_context} is filled
by the caller: empty string in Slice 5, real KB retrieval results in Slice 6.
"""

SAFETY_ANALYSIS = """You are reviewing a security clip from a room safety monitoring system.

Past similar incidents from our knowledge base:
{kb_context}

Looking at these frames, is anyone in danger? If yes, what is happening and why?

Respond in this strict format:
INCIDENT_TYPE: [fall|fight|overcrowding|inactivity|none]
CONFIDENCE: [0.0-1.0]
RATIONALE: [1-2 sentences describing what you observed]"""
