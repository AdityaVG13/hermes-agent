"""The Context Record has two homes and they must not drift.

The plugin renders the record at runtime; this checkout carries a copy the forensics tools
import. A change ported to one and not the other makes post-mortems measure a record no
session ever wrote, which is how the missing-constraints defect stayed invisible.
"""
import os
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "forensics" / "context_record.py"
PEER = Path(os.environ.get(
    "OPTCHAT_RECORD_PY",
    Path.home() / "Developer" / "hermes-plugins" / "optchat" / "record.py",
))


class RecordCopies(unittest.TestCase):
    @unittest.skipUnless(PEER.exists(), f"plugin checkout not present at {PEER}")
    def test_the_runtime_record_and_the_forensics_copy_are_byte_identical(self):
        self.assertEqual(
            HERE.read_bytes(), PEER.read_bytes(),
            "context_record.py drifted from optchat/record.py; port the change to both",
        )
