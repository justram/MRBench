import unittest
import os
from src.utils.logger import Logger

class TestLogger(unittest.TestCase):

    def setUp(self):
        self.log_dir = "./logs"
        self.logger = Logger(log_dir=self.log_dir)

    def tearDown(self):
        if os.path.exists(self.log_dir):
            for f in os.listdir(self.log_dir):
                os.remove(os.path.join(self.log_dir, f))
            os.rmdir(self.log_dir)

    def test_log(self):
        self.logger.log("This is a test log.")
        log_file = os.path.join(self.log_dir, "log.txt")
        self.assertTrue(os.path.exists(log_file))

        with open(log_file, "r") as f:
            content = f.read()
        self.assertIn("This is a test log.", content)

if __name__ == '__main__':
    unittest.main()
