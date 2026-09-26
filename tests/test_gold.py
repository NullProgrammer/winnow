from __future__ import annotations

from piiclf.gold import allocate, stratum_of


def test_cli_imports():
    """Smoke test. All 47 earlier tests passed while cli.py was broken with a
    NameError, because nothing imported it."""
    from piiclf.cli import app

    assert app is not None


class TestStratumOf:
    def test_provider_detectors_collapse_to_one_stratum(self):
        assert stratum_of("aws-access-key", True) == stratum_of("github-pat", True)
        assert stratum_of("aws-access-key", True) == "provider-key|kept"

    def test_non_provider_detectors_stay_separate(self):
        assert stratum_of("at-handle", True) != stratum_of("username-assignment", True)

    def test_survival_splits_the_stratum(self):
        assert stratum_of("ipv4", True) != stratum_of("ipv4", False)
        assert stratum_of("ipv4", False).endswith("|dropped")


class TestAllocate:
    def test_equal_allocation_not_proportional(self):
        # The point of equal allocation: a huge stratum must not eat the budget
        # and leave small ones unmeasurable.
        strata = {"huge": list(range(5000)), "small": list(range(20))}
        q = allocate(strata, budget=40)
        assert q["small"] >= 15
        assert q["huge"] <= 25

    def test_never_exceeds_stratum_size(self):
        strata = {"a": [1, 2, 3], "b": list(range(100))}
        q = allocate(strata, budget=50)
        assert q["a"] <= 3

    def test_small_strata_taken_as_census(self):
        strata = {"tiny": [1], "big": list(range(500))}
        q = allocate(strata, budget=30)
        assert q["tiny"] == 1

    def test_budget_respected(self):
        strata = {f"s{i}": list(range(50)) for i in range(10)}
        q = allocate(strata, budget=100)
        assert sum(q.values()) == 100

    def test_budget_larger_than_population_takes_everything(self):
        strata = {"a": [1, 2], "b": [3, 4, 5]}
        q = allocate(strata, budget=1000)
        assert sum(q.values()) == 5

    def test_empty_strata_dropped(self):
        strata = {"a": [], "b": list(range(10))}
        q = allocate(strata, budget=10)
        assert "a" not in q
