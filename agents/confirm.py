"""Shared yes/no detection for human-in-the-loop confirmations."""
import re

_YES = re.compile(r"^\s*(y|yes|yeah|yep|yup|sure|ok|okay|please|go ahead|do it|confirm(ed)?|"
                  r"create (it|one|a ticket)|sounds good|correct|try again|retry)\b", re.I)
_NO = re.compile(r"^\s*(n|no|nope|nah|not now|no thanks|don't|do not|cancel|never ?mind|stop)\b", re.I)


def is_yes(text: str) -> bool:
    return bool(_YES.search(text)) and not is_no(text)


def is_no(text: str) -> bool:
    return bool(_NO.search(text))
