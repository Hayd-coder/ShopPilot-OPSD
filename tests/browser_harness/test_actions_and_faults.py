import pytest

from browser_harness.actions import match_legal_action, parse_action
from browser_harness.faults import FaultKind, FaultPlan


def test_parse_and_match_actions() -> None:
    search = parse_action(" search[black running shoes] ")
    assert search.kind == "search"
    assert search.argument == "black running shoes"
    assert match_legal_action(search, ("search[<your query>]",)) == search.normalized

    click = parse_action("CLICK[Buy Now]")
    assert click.has_side_effect
    assert match_legal_action(click, ("click[buy now]",)) == "click[buy now]"


@pytest.mark.parametrize("action", ["buy now", "search[]", "click[]", "type[query]"])
def test_rejects_malformed_actions(action: str) -> None:
    with pytest.raises(ValueError):
        parse_action(action)


def test_fault_plan_is_deterministic_and_one_shot() -> None:
    plan = FaultPlan.from_spec("rate_limit=1,4;login=2;timeout=3")
    assert plan.take(1) == FaultKind.RATE_LIMIT
    assert plan.take(1) is None
    assert plan.take(2) == FaultKind.LOGIN_EXPIRED
    assert plan.take(3) == FaultKind.TIMEOUT
    assert plan.take(4) == FaultKind.RATE_LIMIT


def test_fault_plan_rejects_invalid_spec() -> None:
    with pytest.raises(ValueError):
        FaultPlan.from_spec("unknown=1")
    with pytest.raises(ValueError):
        FaultPlan.from_spec("rate_limit=0")
