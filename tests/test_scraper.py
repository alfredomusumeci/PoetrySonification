""" A tests utility for the functions utilised during scraping. """
import unittest
from scraper import *


class TestScrapingUtilities(unittest.TestCase):
    """ Test the scraping utilities. """

    def test_validate_pair_successful(self):
        """ Test that two strings with the same lengths return true """

        notes_line = 'C-D-E A-B *C ^C _D-^E'
        words_line = 'this is a simple test'
        self.assertTrue(validate_pair(notes_line, words_line))

    def test_validate_pair_unsuccessful(self):
        """ Test that two strings with different lengths return false """

        notes_line = 'C-D-E A-B *C ^C _D-^E A'
        words_line = 'this is a simple test'
        self.assertFalse(validate_pair(notes_line, words_line))

    def test_validate_pair_empty(self):
        """ Test that two empty strings return false """

        notes_line = ''
        words_line = ''
        self.assertFalse(validate_pair(notes_line, words_line))

    def test_validate_pair_none(self):
        """ Test that two None strings return false """

        notes_line = None
        words_line = None
        self.assertFalse(validate_pair(notes_line, words_line))

    def test_split_line_successful(self):
        """ Test that a string is split into a list of words by
         1. removing punctuation;
         2. making all letters lowercase;
         3. splitting the sentence into words. """

        words_line = 'ThIs Is, A. SiMple,,, tesT!!!!!'
        self.assertEqual(split_line(words_line), ['this', 'is', 'a', 'simple', 'test'])

    def test_split_line_empty(self):
        """ Test that an empty string returns an empty list """

        words_line = ''
        self.assertEqual(split_line(words_line), [])

    def test_split_into_sentences_odd_length(self):
        """ Test that a string is split into a list of sentences increasing
        by a factor of n words at a time (odd length) """

        words_line = split_line('This is a simple test. This is another sentence.')
        split = split_into_sentences(words_line, 2)
        self.assertEqual(len(split), 5)

    def test_split_into_sentences_even_length(self):
        """ Test that a string is split into a list of sentences increasing
        by a factor of n words at a time (even length) """

        words_line = split_line('This is a simple test. This is another sentence. Hello.')
        split = split_into_sentences(words_line, 2)
        self.assertEqual(len(split), 6)  # last word is included in the previous sentence because even

    def test_parse_note_line_no_wrong_chars(self):
        """ Test that a string with notes and no whitespaces is split correctly"""

        note_line = 'C-D-EA-B*C^C_D-^E.A-BbBbF#'
        self.assertEqual(parse_notes_line(note_line), "C-D-E A-B *C ^C _D-^E .A-Bb Bb F#")

    def test_parse_note_line_with_wrong_chars(self):
        """ Test that a string with notes and no whitespaces is split correctly"""

        note_line = 'C(((((-(D(-(E(A-B*C^C_D-^E.A-Bb(BbF#'
        self.assertEqual(parse_notes_line(note_line), "C-D-E A-B *C ^C _D-^E .A-Bb Bb F#")


if __name__ == '__main__':
    unittest.main()
