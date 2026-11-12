from src.metrics_qa import exact_match, normalize_answer, token_f1


def test_normalize_answer_matches_squad_conventions() -> None:
    assert normalize_answer("The Answer, is here!") == "answer is here"


def test_exact_match_ignores_case_articles_and_punctuation() -> None:
    assert exact_match("The Paris!", "paris") == 1.0
    assert exact_match("Paris", "London") == 0.0


def test_token_f1_uses_multiset_overlap() -> None:
    assert token_f1("the red red fox", "red fox") == 1.0
    assert token_f1("blue whale", "red fox") == 0.0
    assert token_f1("", "") == 1.0
