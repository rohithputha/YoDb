"""The .env reader."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from yodb.dotenv import load_dotenv


class DotenvTests(unittest.TestCase):
    def test_values_are_read_quoted_or_not_and_the_environment_wins(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text('# comment\nYODB_TEST_A=plain value\nexport YODB_TEST_B="quoted = value"\nYODB_TEST_C=\n\nbroken line\n')
            os.environ.pop("YODB_TEST_A", None)
            os.environ["YODB_TEST_B"] = "already"
            try:
                loaded = load_dotenv(path)
                self.assertEqual(loaded, {"YODB_TEST_A": "plain value", "YODB_TEST_C": ""})
                self.assertEqual(os.environ["YODB_TEST_A"], "plain value")
                self.assertEqual(os.environ["YODB_TEST_B"], "already")
                self.assertEqual(load_dotenv(path, override=True)["YODB_TEST_B"], "quoted = value")
            finally:
                for name in ("YODB_TEST_A", "YODB_TEST_B", "YODB_TEST_C"):
                    os.environ.pop(name, None)

    def test_a_missing_file_is_nothing(self) -> None:
        self.assertEqual(load_dotenv("/nonexistent/.env"), {})


if __name__ == "__main__":
    unittest.main()
