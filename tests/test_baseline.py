"""Every value in this file is synthetic. No real credential appears here."""

from __future__ import annotations

from piiclf.baseline import detect
from piiclf.entities import Entity
from piiclf.report import mask

# Synthetic, shape-valid, non-functional.
FAKE_AWS = "AKIAQ7X4MZLP2VNRT8KD"
FAKE_GH = "ghp_R8tLmQ3vXz7NbKdW5yHcF2sJpA9gTe4uZnQx"


def entities(findings) -> set[Entity]:
    return {f.entity for f in findings}


def values(findings, entity: Entity) -> set[str]:
    return {f.value for f in findings if f.entity is entity}


class TestProviderKeys:
    def test_aws_access_key(self):
        src = f'const accessKeyID = "{FAKE_AWS}"\n'
        assert FAKE_AWS in values(detect(src, "main.go"), Entity.KEY)

    def test_github_pat(self):
        src = f'token := "{FAKE_GH}"\n'
        assert FAKE_GH in values(detect(src, "main.go"), Entity.KEY)

    def test_pem_block(self):
        src = "var k = `-----BEGIN RSA PRIVATE KEY-----`\n"
        assert Entity.KEY in entities(detect(src, "main.go"))

    def test_provider_key_survives_test_path_at_lower_confidence(self):
        src = f'const accessKeyID = "{FAKE_AWS}"\n'
        found = [f for f in detect(src, "internal/foo_test.go") if f.entity is Entity.KEY]
        assert found and found[0].confidence < 0.9


class TestSuppression:
    def test_aws_doc_example_key_suppressed(self):
        # AWS's published documentation key contains "EXAMPLE".
        src = 'const accessKeyID = "AKIAIOSFODNN7EXAMPLE"\n'
        assert Entity.KEY not in entities(detect(src, "main.go"))

    def test_placeholder_password(self):
        for ph in ("changeme", "YOUR_PASSWORD", "password", "xxxxxxxx", "${DB_PASS}"):
            src = f'password := "{ph}"\n'
            assert Entity.PASSWORD not in entities(detect(src, "main.go")), ph

    def test_env_var_name_is_not_a_key(self):
        src = 'apiKey := os.Getenv("STRIPE_SECRET_KEY")\n'
        assert Entity.KEY not in entities(detect(src, "main.go"))

    def test_git_sha_is_not_a_key(self):
        src = 'buildSecret := "3f786850e387550fdab836ed7e6dc881de23001b"\n'
        assert Entity.KEY not in entities(detect(src, "main.go"))

    def test_uuid_is_not_a_key(self):
        src = 'secret := "7c9e6679-7425-40de-944b-e07fc1f90ae7"\n'
        assert Entity.KEY not in entities(detect(src, "main.go"))

    def test_placeholder_email_domains(self):
        for dom in ("example.com", "test.com", "yourdomain.com"):
            src = f'// contact: jane@{dom}\n'
            assert Entity.EMAIL not in entities(detect(src, "main.go")), dom

    def test_reserved_ips(self):
        for ip in ("10.0.0.1", "127.0.0.1", "192.168.1.1", "172.16.5.4", "0.0.0.0", "224.0.0.1"):
            src = f'host := "{ip}"\n'
            assert Entity.IP not in entities(detect(src, "main.go")), ip

    def test_documentation_ip_ranges_suppressed(self):
        # RFC 5737 and RFC 2544 ranges exist so docs and tests have addresses
        # that route nowhere. Flagging them would flag every good example.
        for ip in ("192.0.2.1", "198.51.100.7", "203.0.113.45", "198.18.0.1"):
            src = f'host := "{ip}"\n'
            assert Entity.IP not in entities(detect(src, "main.go")), ip

    def test_license_header_orgs_are_not_names(self):
        # Found by scanning the Go source tree: 10,505 NAME hits, every one
        # "The Go Authors" from a copyright header.
        for org in (
            "// Copyright 2009 The Go Authors. All rights reserved.",
            "// Copyright 2016 The Kubernetes Authors",
            "// Copyright (c) 2020 Google Inc",
            "// Copyright 2018 The Apache Software Foundation",
            "// Copyright 2021 Acme Technologies",
            "// Author: The Prometheus Team",
        ):
            assert Entity.NAME not in entities(detect(org + "\n", "main.go")), org

class TestPhase2Fixes:
    """Seven false-positive classes measured on the eval corpus gold sample.

    Together these accounted for 56 of 70 suppression survivors being wrong.
    """

    def test_doc_comment_about_an_author_field_is_not_a_name(self):
        # The `(?i)` flag made [A-Z][a-z]+ match lowercase, so prose matched.
        for line in (
            "// Author represents an author",
            "// Author contains the commit author information",
            "// Author is the GitHub/Gitea user who authored the commit",
            "// author cannot approve their own PR, so it is waived",
        ):
            assert Entity.NAME not in entities(detect(line + "\n", "main.go")), line

    def test_copyright_name_stops_before_lowercase_and(self):
        src = "// Copyright 2015 Matthew Holt and The Caddy Authors\n"
        assert "Matthew Holt" in values(detect(src, "main.go"), Entity.NAME)

    def test_snmp_oids_are_not_ips(self):
        for line in ('oid: ".1.3.6.1.6.3.1.1.4.1.0"', 'x := ".1.0.0.1.3"', 'n := ".9.1.1.1.6"'):
            assert Entity.IP not in entities(detect(line + "\n", "main.go")), line

    def test_plain_ip_still_found_next_to_a_port(self):
        assert "93.184.216.34" in values(detect('h := "93.184.216.34:8080"\n', "a.go"), Entity.IP)

    def test_tsql_variables_are_not_handles(self):
        for line in (
            "DECLARE @ErrorMessage AS nvarchar(500) = 'oops'",
            "SET @Tables += N'x'",
            "INSERT INTO @PCounters SELECT * FROM PerfCounters;",
        ):
            assert Entity.USERNAME not in entities(detect(line + "\n", "q.go")), line

    def test_real_handle_in_prose_still_found(self):
        src = "// taken from a playground link, authored by @awilliams.\n"
        assert "awilliams" in values(detect(src, "main.go"), Entity.USERNAME)

    def test_url_assigned_to_password_var_is_not_a_password(self):
        src = 'passwordURL = "https://api.pwnedpasswords.com/range/"\n'
        assert Entity.PASSWORD not in entities(detect(src, "pwn.go"))

    def test_snake_case_filename_is_not_a_key(self):
        src = 'const authorizedPrincipalsFile = "authorized_principals"\n'
        assert Entity.KEY not in entities(detect(src, "ssh.go"))

    def test_pem_in_test_file_suppressed_but_issued_key_kept(self):
        pem = 'k := `-----BEGIN OPENSSH PRIVATE KEY-----`\n'
        assert Entity.KEY not in entities(detect(pem, "models/ssh_key_test.go"))
        # An issued credential in a test file is still a leak worth flagging.
        assert Entity.KEY in entities(detect(f'k := "{FAKE_AWS}"\n', "models/ssh_key_test.go"))

    def test_underscored_version_key_is_not_an_ip(self):
        assert Entity.IP not in entities(detect("rubygems_version: 2.7.6.2\n", "m.go"))

    def test_conversion_does_not_suppress_a_real_ip(self):
        src = 'addr := "93.184.216.34" // conversion helper\n'
        assert "93.184.216.34" in values(detect(src, "m.go"), Entity.IP)

    def test_github_noreply_domain_suppressed(self):
        src = '"email": "baxterthehacker@users.noreply.github.com",\n'
        assert Entity.EMAIL not in entities(detect(src, "hook.go"))

    def test_pem_outside_tests_still_found(self):
        assert Entity.KEY in entities(detect("k := `-----BEGIN RSA PRIVATE KEY-----`\n", "k.go"))


class TestOtherEntities2:
    def test_real_person_in_copyright_still_found(self):
        src = "// Copyright (c) 2024 Jane Roe\n"
        assert "Jane Roe" in values(detect(src, "main.go"), Entity.NAME)

    def test_version_string_is_not_an_ip(self):
        src = 'const Version = "1.24.3.1"\n'
        assert Entity.IP not in entities(detect(src, "main.go"))

    def test_secrets_dropped_in_test_files(self):
        src = 'password := "Tr0ub4dor3xK"\n'
        assert Entity.PASSWORD not in entities(detect(src, "internal/svc_test.go"))

    def test_graphql_literal_is_not_a_key(self):
        # This FP class was found by running the baseline over real Go: a long
        # query/mutation string assigned to a key-ish identifier passes the
        # entropy test because prose sits near 4.0 bits/char.
        src = (
            'authQuery := "mutation UpsertCustomer($input: CustomerInput!) '
            '{ upsertCustomer(input: $input) { id sourceId sourceType firstName } }"\n'
        )
        assert Entity.KEY not in entities(detect(src, "queries.go"))

    def test_url_is_not_a_key(self):
        src = 'tokenURL := "https://auth.example-vendor.io/oauth2/v1/token/exchange"\n'
        assert Entity.KEY not in entities(detect(src, "client.go"))

    def test_sql_literal_is_not_a_key(self):
        src = 'secretQuery := "SELECT id, source_id FROM customers WHERE source_type = $1"\n'
        assert Entity.KEY not in entities(detect(src, "repo.go"))

    def test_key_with_whitespace_rejected_but_password_allows_passphrase(self):
        assert Entity.KEY not in entities(detect('apiKey := "Xk92 LmQp 4Zt8 Nv3W"\n', "a.go"))
        assert Entity.PASSWORD in entities(detect('password := "correct horse battery staple"\n', "a.go"))

    def test_raw_mode_keeps_what_suppression_drops(self):
        src = 'password := "changeme"\n'
        assert Entity.PASSWORD not in entities(detect(src, "main.go"))
        assert Entity.PASSWORD in entities(detect(src, "main.go", apply_suppression=False))


class TestOtherEntities:
    def test_real_looking_email(self):
        src = "// maintainer reachable at jane.doe@acmecorp.io\n"
        assert "jane.doe@acmecorp.io" in values(detect(src, "main.go"), Entity.EMAIL)

    def test_public_ip(self):
        src = 'upstream := "93.184.216.34:8080"\n'
        assert "93.184.216.34" in values(detect(src, "main.go"), Entity.IP)

    def test_dsn_splits_user_and_password(self):
        src = 'dsn := "postgres://svc_reader:Xk92LmQp4Zt@db.internal:5432/offers"\n'
        found = detect(src, "main.go")
        assert "Xk92LmQp4Zt" in values(found, Entity.PASSWORD)
        assert "svc_reader" in values(found, Entity.USERNAME)

    def test_dsn_password_not_also_reported_as_email(self):
        # Regression: `user:pass@host` matches the email pattern, and EMAIL's
        # mask exposes the first character — leaking a character of the password
        # that PASSWORD's mask deliberately hides.
        src = 'dsn := "postgres://svc_reader:Xk92LmQp4Zt@db.internal:5432/offers"\n'
        found = detect(src, "main.go")
        assert Entity.EMAIL not in entities(found)
        assert "Xk92LmQp4Zt" in values(found, Entity.PASSWORD)

    def test_real_email_still_found_alongside_a_dsn(self):
        src = (
            '// contact jane.doe@acmecorp.io\n'
            'dsn := "postgres://svc_reader:Xk92LmQp4Zt@db.internal:5432/offers"\n'
        )
        assert "jane.doe@acmecorp.io" in values(detect(src, "main.go"), Entity.EMAIL)

    def test_author_comment_name(self):
        src = "// Author: Firstname Lastname\n"
        assert "Firstname Lastname" in values(detect(src, "main.go"), Entity.NAME)

    def test_copyright_name(self):
        src = "// Copyright (c) 2024 Jane Roe\n"
        assert "Jane Roe" in values(detect(src, "main.go"), Entity.NAME)

    def test_godoc_tags_are_not_usernames(self):
        src = "// @param ctx context\n// @return error\n"
        assert Entity.USERNAME not in entities(detect(src, "main.go"))


class TestPositions:
    def test_line_and_column_are_one_based(self):
        src = f'package main\n\nvar k = "{FAKE_AWS}"\n'
        f = next(x for x in detect(src, "main.go") if x.entity is Entity.KEY)
        assert f.line == 3
        assert src.splitlines()[f.line - 1][f.column - 1 :].startswith(FAKE_AWS)


class TestMasking:
    def test_mask_never_contains_full_value(self):
        cases = [
            (Entity.KEY, FAKE_AWS),
            (Entity.KEY, FAKE_GH),
            (Entity.PASSWORD, "Xk92LmQp4Zt"),
            (Entity.EMAIL, "jane.doe@acmecorp.io"),
            (Entity.IP, "93.184.216.34"),
            (Entity.NAME, "Firstname Lastname"),
            (Entity.USERNAME, "svc_reader"),
        ]
        for entity, value in cases:
            masked = mask(entity, value)
            assert value not in masked, (entity, masked)

    def test_key_mask_keeps_provider_prefix_for_rotation(self):
        assert mask(Entity.KEY, FAKE_AWS).startswith("AKIA")
        assert mask(Entity.KEY, FAKE_GH).startswith("ghp_")

    def test_password_mask_reveals_only_length(self):
        assert mask(Entity.PASSWORD, "Xk92LmQp4Zt") == "…[11 chars]"

    def test_ip_mask_keeps_first_octet_only(self):
        assert mask(Entity.IP, "93.184.216.34") == "93.x.x.x"
