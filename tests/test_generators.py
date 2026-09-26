from __future__ import annotations

import random

from piiclf.baseline import _is_reserved_ip
from piiclf.generators import NEGATIVES, POSITIVES, VOCAB, generate


def rng(seed: int = 1) -> random.Random:
    return random.Random(seed)


class TestVocabularyHoldout:
    def test_train_and_eval_vocabularies_are_disjoint(self):
        # Rows overlapping would be fine; vocabularies overlapping would let
        # memorisation pass for generalisation.
        for field in VOCAB["train"]:
            a, b = set(VOCAB["train"][field]), set(VOCAB["eval"][field])
            assert not (a & b), f"{field} overlaps: {a & b}"

    def test_generated_emails_use_only_their_split_domains(self):
        r = rng()
        for split in ("train", "eval"):
            other = set(VOCAB["eval" if split == "train" else "train"]["domains"])
            for _ in range(200):
                v = POSITIVES["EMAIL"](r, VOCAB[split])
                assert v.text.split("@")[1] not in other

    def test_generated_keys_use_only_their_split_prefixes(self):
        r = rng()
        for split in ("train", "eval"):
            other = VOCAB["eval" if split == "train" else "train"]["key_prefixes"]
            for _ in range(200):
                v = POSITIVES["KEY"](r, VOCAB[split])
                assert not any(v.text.startswith(p) for p in other)


class TestPositivesFollowTheRubric:
    def test_name_positive_is_data_not_attribution(self):
        r = rng()
        for _ in range(100):
            v = POSITIVES["NAME"](r, VOCAB["train"])
            assert v.entity == "NAME"
            low = v.text.lower()
            assert "copyright" not in low and "author" not in low
            assert "comment" not in v.kinds  # attribution positions are negatives

    def test_username_positive_is_an_account_not_a_handle(self):
        r = rng()
        for _ in range(100):
            v = POSITIVES["USERNAME"](r, VOCAB["train"])
            assert not v.text.startswith("@")

    def test_ip_positive_is_never_reserved_or_documentation(self):
        r = rng()
        for _ in range(500):
            v = POSITIVES["IP"](r, VOCAB["train"])
            assert not _is_reserved_ip([int(o) for o in v.text.split(".")]), v.text

    def test_key_positive_has_no_whitespace(self):
        r = rng()
        for _ in range(200):
            assert not any(c.isspace() for c in POSITIVES["KEY"](r, VOCAB["train"]).text)

    def test_passphrase_passwords_may_contain_spaces(self):
        r = rng()
        texts = [POSITIVES["PASSWORD"](r, VOCAB["train"]).text for _ in range(300)]
        assert any(" " in t for t in texts), "expected some passphrases"

    def test_every_entity_class_is_generable(self):
        r = rng()
        assert {POSITIVES[k](r, VOCAB["train"]).entity for k in POSITIVES} == {
            "EMAIL", "KEY", "PASSWORD", "NAME", "USERNAME", "IP"
        }


class TestHardNegatives:
    def test_all_negatives_carry_no_entity(self):
        r = rng()
        for fn in NEGATIVES:
            v = fn(r, VOCAB["train"])
            assert v.entity is None, f"{fn.__name__} produced a positive"
            assert v.why, f"{fn.__name__} has no provenance"

    def test_attribution_is_a_negative_including_real_people(self):
        r = rng()
        texts = [fn(r, VOCAB["train"]).text for fn in NEGATIVES for _ in range(5)]
        joined = " ".join(texts).lower()
        assert "copyright" in joined
        assert "author" in joined

    def test_negative_classes_cover_the_measured_fp_list(self):
        r = rng()
        whys = {fn(r, VOCAB["train"]).why for fn in NEGATIVES}
        # Each of these was a false positive the Phase 2 gold set actually caught.
        for expected in (
            "org-attribution", "person-attribution", "public-handle", "placeholder-email",
            "env-var-name", "git-sha", "uuid", "snake-case-filename", "url-as-secret",
            "tsql-variable", "snmp-oid", "reserved-or-doc-ip", "version-string",
            "service-default", "app-display-name", "constant-identifier", "pem-test-fixture",
        ):
            assert expected in whys, f"missing measured FP class: {expected}"


class TestGenerateMix:
    def test_negative_rate_is_respected(self):
        r = rng()
        vals = [generate(r, "train", negative_rate=0.4) for _ in range(4000)]
        frac = sum(1 for v in vals if v.entity is None) / len(vals)
        assert 0.36 < frac < 0.44, frac

    def test_all_zero_and_all_one_rates(self):
        r = rng()
        assert all(generate(r, "train", 0.0).entity for _ in range(200))
        assert all(generate(r, "train", 1.0).entity is None for _ in range(200))

    def test_every_value_declares_at_least_one_site_kind(self):
        r = rng()
        for _ in range(1000):
            v = generate(r, "train")
            assert v.kinds and all(k in ("string", "raw_string", "comment") for k in v.kinds)

    def test_deterministic_for_a_given_seed(self):
        a = [generate(random.Random(99), "train").text for _ in range(1)]
        b = [generate(random.Random(99), "train").text for _ in range(1)]
        assert a == b
