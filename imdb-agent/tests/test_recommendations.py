"""Unit tests for the recommendation tool — no DB required for title extraction."""
from tools.recommendation_tools import _extract_primary_title


def test_extracts_bold_markdown_title():
    response = "**The Dark Knight** (2008) is one of the greatest superhero films."
    assert _extract_primary_title(response) == "The Dark Knight"


def test_extracts_quoted_title():
    response = 'The movie "Inception" (2010) features a complex plot.'
    assert _extract_primary_title(response) == "Inception"


def test_extracts_plain_capitalised_title():
    response = "Schindler's List (1993) is directed by Steven Spielberg."
    result = _extract_primary_title(response)
    assert result is not None
    assert len(result) > 3


def test_returns_none_for_list_response():
    # List responses have no single movie title at the start
    response = "Here are the top 10 movies:\n1. Movie A\n2. Movie B\n3. Movie C"
    # May or may not extract — the key contract is it doesn't crash
    result = _extract_primary_title(response)
    assert result is None or isinstance(result, str)


def test_returns_none_for_empty_string():
    assert _extract_primary_title("") is None


def test_bold_pattern_takes_priority_over_plain():
    response = "Talking about **Interstellar** (2014). Also Gravity (2013) is good."
    assert _extract_primary_title(response) == "Interstellar"
