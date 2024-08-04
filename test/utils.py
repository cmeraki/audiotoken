import os
import pathlib
import unittest
from audiotoken.utils import sanitize_path

class TestSanitizePath(unittest.TestCase):
    def setUp(self):
        self.test_dir = pathlib.Path(__file__).parent / "test_dir"
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        # TODO: Remove test directory
        pass

    def test_relative_to_absolute(self):
        relative_path = "test_file.txt"
        sanitized = sanitize_path(relative_path)
        self.assertTrue(os.path.isabs(sanitized))

    def test_expand_user_path(self):
        user_path = "~/test_file.txt"
        sanitized = sanitize_path(user_path)
        self.assertNotIn("~", sanitized)
        self.assertTrue(sanitized.startswith(os.path.expanduser("~")))

    def test_create_directory(self):
        new_dir = self.test_dir / "new_directory"
        sanitized = sanitize_path(new_dir)
        self.assertTrue(os.path.exists(sanitized))


if __name__ == "__main__":
    unittest.main()
