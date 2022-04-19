""" A tests utility for the baseline.py file. """
from baseline import *
from music_parser import generate_midi_notes
from music21 import instrument
import unittest


class TestBaselineUtilities(unittest.TestCase):
    """ Test the baseline utilities. """

    melody = generate_random_melody(10)

    def test_generate_random_melody_with_length(self):
        """ Test that the random melody generated has pre-established length. """

        self.assertEqual(len(self.melody), 10)

    def test_random_melody_is_acceptable(self):
        """ Test that the random melody generated is is in the correct format. """

        try:
            generate_midi_notes(self.melody, _instrument=instrument.Piano(), offset=0)
        except ValueError:
            self.fail("Melody is not in the correct format.")


if __name__ == '__main__':
    unittest.main()
