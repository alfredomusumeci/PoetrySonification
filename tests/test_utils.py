""" A test utility for the utils functions; additional test cases can be added when new functions are created. """
import unittest
from utils import *


class TestUtils(unittest.TestCase):
    """ Test cases for the utils functions. """

    def test_is_hexadecimal(self):
        """ Test is_hexadecimal function by converting a random string to
        an integer with base 16"""

        self.assertIsInstance(int(generate_random_string(16), 16), int)
        self.assertIsInstance(int(generate_random_string(16), 16), int)
        self.assertIsInstance(int(generate_random_string(16), 16), int)

    def test_delete_file(self):
        """ Test delete_file function by creating a file and then deleting it. """

        with open('test.txt', 'w') as f:
            f.write('If this file is not destroyed, the test failed.')

        delete_file('test.txt')
        self.assertFalse(os.path.exists('test.txt'))


if __name__ == '__main__':
    unittest.main()
