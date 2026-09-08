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
