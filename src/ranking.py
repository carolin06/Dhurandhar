# src/ranking.py
def score_document(doc, verification_prob, emergency=False):
    base = verification_prob * doc.get("trust", 0.5)

    pogo = doc.get("pogo", 0)
    base *= (1 - min(0.5, 0.05 * pogo))

    if emergency:
        base *= 1.3

    return base
