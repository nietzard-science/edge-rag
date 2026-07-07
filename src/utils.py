"""
Shared utilities for the Edge-RAG system.

Centralises small, stateless helper functions used by multiple layers so
that identical implementations are not duplicated across modules.
"""


def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Word-level Jaccard similarity between two text strings.

    similarity(A, B) = |A ∩ B| / |A ∪ B|

    where A and B are the lower-cased word-token sets of each text.

    Reference:
        Jaccard, P. (1901). Étude comparative de la distribution florale dans
        une portion des Alpes et du Jura. Bull. Soc. Vaudoise Sci. Nat., 37,
        241–272. DOI: 10.5169/seals-266450
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    union = len(words1 | words2)
    return len(words1 & words2) / union if union > 0 else 0.0
