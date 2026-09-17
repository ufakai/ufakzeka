"""Turkish XCOPA prompt: premise followed by a connective, choices as continuations."""

CONNECTIVES = {"cause": "çünkü", "effect": "bu yüzden"}


def _lower_first(s: str) -> str:
    return s[0].lower() + s[1:] if s else s


def xcopa_doc_to_text(doc):
    premise = doc["premise"].rstrip(".")
    return f"{premise} {CONNECTIVES[doc['question']]}"


def xcopa_doc_to_choice(doc):
    return [" " + _lower_first(doc["choice1"]), " " + _lower_first(doc["choice2"])]


def arc_doc_to_target(doc):
    """ARC answer keys are letters or digits depending on the item; map both to an index."""
    labels = doc["choices"]["label"]
    key = doc["answerKey"]
    if key in labels:
        return labels.index(key)
    if key.isdigit():
        return int(key) - 1
    return "ABCDE".index(key)
