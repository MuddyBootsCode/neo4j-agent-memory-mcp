"""Labels must not outlive the query they were made against (MUD-460).

step4 resumes on the key ``"<query_id>:<lesson_id>"``, so a query whose
text changes while its id stays the same is skipped as already labelled
and scored against ground truth built for the old text. That is silent:
nothing errors, the numbers are simply wrong. The error-keyed set made it
concrete — regenerating it after the truncation fix rewrites 34 of 68
prompts under their original ids.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments", "golden"))


class TestQueryFingerprint:
    def test_the_same_query_fingerprints_the_same(self):
        from lib import query_fingerprint

        q = {"query_id": 1, "prompt": "boom", "files": ["b.py", "a.py"]}
        assert query_fingerprint(q) == query_fingerprint(
            {"query_id": 99, "prompt": "boom", "files": ["a.py", "b.py"]}
        )

    def test_changed_text_changes_the_fingerprint(self):
        from lib import query_fingerprint

        short = {"query_id": 1, "prompt": "Traceback ...", "files": []}
        full = {"query_id": 1, "prompt": "Traceback ... ValueError: boom", "files": []}
        assert query_fingerprint(short) != query_fingerprint(full)

    def test_changed_file_context_changes_it_too(self):
        """The labeller sees the files, so they are part of what was judged."""
        from lib import query_fingerprint

        a = {"query_id": 1, "prompt": "boom", "files": ["a.py"]}
        b = {"query_id": 1, "prompt": "boom", "files": ["a.py", "b.py"]}
        assert query_fingerprint(a) != query_fingerprint(b)


class TestStaleLabels:
    def test_only_the_changed_query_loses_its_labels(self):
        from lib import query_fingerprint, stale_label_keys

        unchanged = {"query_id": 1, "prompt": "same", "files": []}
        changed_before = {"query_id": 2, "prompt": "old text", "files": []}
        changed_now = {"query_id": 2, "prompt": "new text", "files": []}
        stored = {
            "1": query_fingerprint(unchanged),
            "2": query_fingerprint(changed_before),
        }
        labels = {"1:aaa": True, "1:bbb": False, "2:aaa": True, "2:bbb": False}

        stale = stale_label_keys(labels, [unchanged, changed_now], stored)

        assert stale == {"2:aaa", "2:bbb"}

    def test_a_run_with_no_fingerprints_yet_keeps_everything(self):
        """Older runs were labelled before fingerprints existed; their
        labels are still valid and must not be thrown away."""
        from lib import stale_label_keys

        labels = {"1:aaa": True}

        assert stale_label_keys(labels, [{"query_id": 1, "prompt": "p", "files": []}], {}) == set()

    def test_a_query_that_is_new_has_nothing_to_invalidate(self):
        from lib import query_fingerprint, stale_label_keys

        known = {"query_id": 1, "prompt": "p", "files": []}
        fresh = {"query_id": 2, "prompt": "q", "files": []}
        stored = {"1": query_fingerprint(known)}

        assert stale_label_keys({"1:aaa": True}, [known, fresh], stored) == set()


class TestLabelCompleteness:
    def test_a_fully_labelled_run_reports_nothing_missing(self):
        from lib import unlabeled_pairs

        queries = [{"query_id": 1, "repo": "r"}, {"query_id": 2, "repo": "r"}]
        pool = [{"id": "a", "repo": "r"}, {"id": "b", "repo": "r"}]
        labels = {f"{q}:{lid}": True for q in (1, 2) for lid in ("a", "b")}

        assert unlabeled_pairs(labels, queries, pool) == 0

    def test_an_interrupted_relabel_is_counted(self):
        """step4 deletes a changed query's labels before calling the API.
        A failure there leaves the query with no judgments at all, and
        step5 would quietly shrink its recall denominator instead."""
        from lib import unlabeled_pairs

        queries = [{"query_id": 1, "repo": "r"}, {"query_id": 2, "repo": "r"}]
        pool = [{"id": "a", "repo": "r"}, {"id": "b", "repo": "r"}]
        labels = {"1:a": True, "1:b": False}

        assert unlabeled_pairs(labels, queries, pool) == 2

    def test_lessons_from_another_repo_are_not_owed_a_label(self):
        from lib import unlabeled_pairs

        queries = [{"query_id": 1, "repo": "r"}]
        pool = [{"id": "a", "repo": "r"}, {"id": "z", "repo": "other"}]

        assert unlabeled_pairs({"1:a": True}, queries, pool) == 0


class TestFingerprintsFollowTheLabels:
    def test_the_first_saved_label_fingerprints_its_query(self):
        """The fingerprint answers "which text were these judged against",
        not "are we finished". Withholding it until a query is complete
        leaves a half-labelled query unfingerprinted, and a text change
        there would mix old verdicts with new ones under one certificate.
        Completeness is checked separately, by unlabeled_pairs."""
        from lib import fingerprints_for_labelled, query_fingerprint

        done = {"query_id": 1, "prompt": "p", "files": []}
        half = {"query_id": 2, "prompt": "q", "files": []}
        untouched = {"query_id": 3, "prompt": "r", "files": []}
        labels = {"1:a": True, "1:b": False, "2:a": True}

        assert fingerprints_for_labelled(labels, [done, half, untouched]) == {
            "1": query_fingerprint(done),
            "2": query_fingerprint(half),
        }

    def test_a_query_changed_mid_labelling_is_caught_on_resume(self):
        """The window the checkpoint rule opened: interrupt after one
        chunk, change the query, resume. The partial labels must be found
        stale rather than topped up against the new text."""
        from lib import fingerprints_for_labelled, stale_label_keys

        before = {"query_id": 1, "prompt": "old text", "files": []}
        after = {"query_id": 1, "prompt": "new text", "files": []}
        partial = {"1:a": True}
        stored = fingerprints_for_labelled(partial, [before])

        assert stale_label_keys(partial, [after], stored) == {"1:a"}


class TestRewriteProvenance:
    def test_a_rewrite_is_stale_when_its_query_changed(self):
        """HyDE guesses are keyed by query_id and step5 attaches them the
        same way, so a regenerated query silently inherits the guesses
        written for the prompt it replaced."""
        from lib import query_fingerprint, stale_rewrites

        before = {"query_id": 1, "prompt": "the deploy fails", "files": []}
        after = {"query_id": 1, "prompt": "the limiter returns 200", "files": []}
        store = {"1": {"fingerprint": query_fingerprint(before), "lessons": ["guess"]}}

        assert stale_rewrites(store, [before]) == set()
        assert stale_rewrites(store, [after]) == {"1"}

    def test_rewrites_written_before_fingerprints_are_reported_unverified(self):
        from lib import rewrite_lessons, stale_rewrites

        legacy = {"1": ["a guess", "another"]}

        assert rewrite_lessons(legacy["1"]) == ["a guess", "another"]
        assert stale_rewrites(legacy, [{"query_id": 1, "prompt": "p", "files": []}]) == set()

    def test_the_lessons_read_the_same_from_either_shape(self):
        from lib import rewrite_lessons

        assert rewrite_lessons({"fingerprint": "abc", "lessons": ["x"]}) == ["x"]
        assert rewrite_lessons(["x"]) == ["x"]
        assert rewrite_lessons(None) == []
