"""Label comparison used when merging views and when matching the clinic list."""

from __future__ import annotations

import re

from rapidfuzz import fuzz

_WORD = re.compile(r"[0-9A-Za-z]+")
_SIDE_VIEW = {"left", "right"}


def token_ratio(left: str, right: str) -> float:
    """rapidfuzz token ratio, 0 to 100."""
    return float(fuzz.token_ratio(left, right))


def match_text(value: str) -> str:
    """Words only, so a question mark does not block a label match."""
    return " ".join(_WORD.findall(value))


def side_view_core(label: str) -> str:
    """Drop a leading left/right when it only marks which photo the view came from."""
    parts = label.split()
    if len(parts) >= 2 and parts[0].casefold().strip(".,:;") in _SIDE_VIEW:
        return " ".join(parts[1:])
    return label


_COLOR_TOKENS = {
    "black",
    "blue",
    "brown",
    "gold",
    "gray",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "white",
    "yellow",
}


def label_query_score(query: str, label: str, drug_name: str | None = None) -> float:
    """Best token score of the question against a label, its side-view core, or drug name."""
    names = [label]
    if drug_name:
        names.append(drug_name)
    core = side_view_core(label)
    if core.casefold() != label.casefold():
        names.append(core)
    return max(token_ratio(match_text(query), match_text(name)) for name in names)


def query_label_close(query: str, label: str, drug_name: str | None, label_sim: float) -> bool:
    """True when a detection could answer the question.

    The score uses the catalog label threshold. A color word in the question
    that contradicts the label is not close: "blue water bottle" must not keep
    a black water bottle just because the other words match.
    """
    if _color_conflict(query, label):
        return False
    return label_query_score(query, label, drug_name) >= label_sim


def _color_conflict(query: str, label: str) -> bool:
    query_colors = set(match_text(query).casefold().split()) & _COLOR_TOKENS
    if not query_colors:
        return False
    label_colors = set(match_text(label).casefold().split()) & _COLOR_TOKENS
    if not label_colors:
        return False
    return query_colors.isdisjoint(label_colors)


def same_label_family(left: str, right: str, label_sim: float) -> bool:
    """True when two labels are the same object at the existing threshold.

    A left view and a right view of one object stay in the family even when
    the side words pull the raw score just under label_sim. Different objects
    stay apart; the threshold itself does not move.
    """
    if token_ratio(left, right) >= label_sim:
        return True
    core_left = side_view_core(left)
    core_right = side_view_core(right)
    if core_left == left and core_right == right:
        return False
    if not core_left or not core_right:
        return False
    return token_ratio(core_left, core_right) >= label_sim
