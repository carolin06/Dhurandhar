from symspellpy import SymSpell, Verbosity
from pathlib import Path

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

DICT_PATH = Path(__file__).parent / "frequency_dictionary_en_82_765.txt"

sym_spell.load_dictionary(
    str(DICT_PATH),
    term_index=0,
    count_index=1
)

PROTECTED_WORDS = {
    "what", "is", "are", "was", "were",
    "who", "why", "how", "when", "where",
    "near", "me", "in", "on", "at",
    "to", "from", "of", "for"
}

def autocorrect_query(query: str) -> str:
    corrected = []

    for word in query.lower().split():
        if word in PROTECTED_WORDS:
            corrected.append(word)
        else:
            suggestions = sym_spell.lookup(
                word,
                Verbosity.CLOSEST,
                max_edit_distance=2
            )
            corrected.append(suggestions[0].term if suggestions else word)

    return " ".join(corrected)
