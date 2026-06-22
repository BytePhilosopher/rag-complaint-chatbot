"""Unit tests for the Task 1 cleaning + product-mapping helpers."""
from src.eda_preprocessing import clean_text, map_product


def test_clean_lowercases_and_strips_special_chars():
    assert clean_text("Hello, WORLD!!! $$$") == "hello world"


def test_clean_removes_xxxx_redactions():
    out = clean_text("On XX/XX/XXXX my account XXXX was charged")
    assert "xx" not in out
    assert "account" in out and "charged" in out


def test_clean_removes_boilerplate_opening():
    out = clean_text("I am writing to file a complaint about a late fee")
    assert "complaint" not in out
    assert "late fee" in out


def test_clean_empty_when_only_noise():
    assert clean_text("XXXX !!! $$$") == ""


def test_map_product_credit_card():
    assert map_product("Credit card or prepaid card") == "Credit Card"
    assert map_product("Credit card") == "Credit Card"


def test_map_product_personal_loan():
    assert map_product("Payday loan, title loan, or personal loan") == "Personal Loan"


def test_map_product_savings():
    assert map_product("Checking or savings account") == "Savings Account"


def test_map_product_money_transfer():
    assert map_product("Money transfer, virtual currency, or money service") == "Money Transfer"


def test_map_product_unmatched_returns_none():
    assert map_product("Mortgage") is None
    assert map_product("Debt collection") is None
