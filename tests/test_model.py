""" A test utility for the model.py file """
from model import *
import unittest


class TestModelUtilities(unittest.TestCase):
    """ Test the model utilities """

    sonification_model = RNNModel()

    def test_token_start_end_addition(self):
        """ Test that the start and end tokens are added to the beginning and end of the output """

        for line in self.sonification_model.notes:
            self.assertTrue('sos ' in line and ' eos' in line)

    def test_notes_contain_no_digit(self):
        """ Test that the notes do not contain any digits """

        for line in self.sonification_model.notes:
            self.assertFalse(any(char.isdigit() for char in line))

    def test_input_text_has_contractions_expanded(self):
        """ Test that contractions are expanded on the input text"""

        text = 'i\'m a test i won\'t do damage'
        self.assertEqual(self.sonification_model.clean_text(text), 'i am a test i will not do damage')

    def test_input_text_is_lowercase(self):
        """ Test that the input text is lowercase """

        text = 'I AM A TEST'
        self.assertEqual(self.sonification_model.clean_text(text), 'i am a test')

    def test_input_text_no_punctuation(self):
        """ Test that the input text does not contain any punctuation """

        text = 'I am a test, I am a test.'
        self.assertEqual(self.sonification_model.clean_text(text), 'i am a test i am a test')

    def test_input_text_no_numbers(self):
        """ Test that the input text does not contain any numbers """

        text = 'I am a test, I am a test. 1, 2, 3'
        self.assertEqual(self.sonification_model.clean_text(text).strip(), 'i am a test i am a test')

    def test_eos_is_removed_from_output(self):
        """ Test that the end of sentence token is removed from the output """

        text = 'C D E F G A B eos'
        self.assertEqual(self.sonification_model.polish_output(text), 'C D E F G A B')


if __name__ == '__main__':
    unittest.main()
