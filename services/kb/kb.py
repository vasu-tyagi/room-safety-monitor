"""pgvector knowledge base (Slice 6).

KB.write stores a prompt+rationale+label embedding and associated metadata.
KB.find_similar retrieves prior incidents by cosine similarity for VLM prompt
grounding.

The text vectorised per entry is: prompt + " " + rationale + " " + label.
Using all three fields captures what triggered analysis (prompt), what was
observed (rationale), and the classification (label), which produces richer
similarity retrieval than prompts alone.
"""
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from services.kb.embedder import embed
from services.kb.models import KBEntryRow


@dataclass
class KBEntry:
    id: uuid.UUID
    prompt: str
    rationale: str
    label: str
    operator_decision: str
    similarity_score: float
    metadata: Dict = field(default_factory=dict)


class KB:
    def __init__(self, session: Session):
        self._session = session

    def write(
        self,
        prompt: str,
        rationale: str,
        label: str,
        operator_decision: str,
        metadata: dict,
    ) -> uuid.UUID:
        """Embed and store a KB entry. Returns the new entry's UUID."""
        text = f"{prompt} {rationale} {label}"
        vec = embed(text)
        entry_id = uuid.uuid4()
        row = KBEntryRow(
            id=entry_id,
            embedding=vec,
            prompt_text=prompt,
            rationale_text=rationale,
            label=label,
            operator_decision=operator_decision,
            meta=metadata,
        )
        self._session.add(row)
        self._session.commit()
        return entry_id

    def find_similar(
        self,
        query_text: str,
        top_k: int = 3,
        threshold: float = 0.7,
    ) -> List[KBEntry]:
        """Return up to top_k entries with cosine similarity >= threshold.

        pgvector <=> is cosine distance (0 = identical, 2 = opposite).
        similarity = 1 - cosine_distance, so threshold=0.7 means distance <= 0.3.
        """
        query_vec = embed(query_text)
        distance_cutoff = 1.0 - threshold

        dist_expr = KBEntryRow.embedding.cosine_distance(query_vec)
        sim_expr = (1 - dist_expr).label("similarity")

        stmt = (
            select(KBEntryRow, sim_expr)
            .where(dist_expr <= distance_cutoff)
            .order_by(dist_expr)
            .limit(top_k)
        )

        results = []
        for row in self._session.execute(stmt):
            entry_row, sim = row[0], float(row[1])
            results.append(KBEntry(
                id=entry_row.id,
                prompt=entry_row.prompt_text,
                rationale=entry_row.rationale_text,
                label=entry_row.label,
                operator_decision=entry_row.operator_decision,
                similarity_score=sim,
                metadata=entry_row.meta or {},
            ))
        return results
