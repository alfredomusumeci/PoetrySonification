""" A test utility for the music_parser.py file """
import os
import unittest
from music21 import pitch

import utils
from music_parser import *
from baseline import generate_random_melody


class TestNoteGeneration(unittest.TestCase):
    """ Test the generation of notes """

    def test_octave2_note_generation(self):
        """ Test that C2 is generated correctly """

        note_string = '_C'  # C2
        note_obj = note.Note('C2')
        self.assertEqual(convert_note_to_midi(note_string), note_obj)

    def test_octave3_note_generation(self):
        """ Test that C3 is generated correctly """

        note_string = '.C'  # C3
        note_obj = note.Note('C3')
        self.assertEqual(convert_note_to_midi(note_string), note_obj)

    def test_octave4_note_generation(self):
        """ Test that C4 is generated correctly """

        note_string = 'C'  # C4
        note_obj = note.Note('C4')
        self.assertEqual(convert_note_to_midi(note_string), note_obj)

    def test_octave5_note_generation(self):
        """ Test that C5 is generated correctly """

        note_string = '^C'  # C5
        note_obj = note.Note('C5')
        self.assertEqual(convert_note_to_midi(note_string), note_obj)

    def test_octave6_note_generation(self):
        """ Test that C6 is generated correctly """

        note_string = '*C'  # C6
        note_obj = note.Note('C6')
        self.assertEqual(convert_note_to_midi(note_string), note_obj)

    def test_sharp_note_generation(self):
        """ Test that C# is generated correctly """

        note_string = 'C#'  # C4#
        note_obj = note.Note('C4#')
        self.assertEqual(convert_note_to_midi(note_string), note_obj)

    def test_flat_note_generation(self):
        """ Test that Bb is generated correctly """

        note_string = 'Bb'  # B4b
        note_obj = note.Note('B4b')
        self.assertEqual(convert_note_to_midi(note_string), note_obj)

    def test_octave2_sharp_note_generation(self):
        """ Test that C#2 is generated correctly """

        note_string = '_C#'  # C2#
        note_obj = note.Note('C2#')
        self.assertEqual(convert_note_to_midi(note_string), note_obj)

    def test_octave3_sharp_note_generation(self):
        """ Test that C#3 is generated correctly """

        note_string = '.C#'  # C3#
        note_obj = note.Note('C3#')
        self.assertEqual(convert_note_to_midi(note_string), note_obj)

    def test_octave5_sharp_note_generation(self):
        """ Test that C#5 is generated correctly """

        note_string = '^C#'  # C5#
        note_obj = note.Note('C5#')
        self.assertEqual(convert_note_to_midi(note_string), note_obj)

    def test_octave6_sharp_note_generation(self):
        """ Test that C#6 is generated correctly """

        note_string = '*C#'  # C6#
        note_obj = note.Note('C6#')
        self.assertEqual(convert_note_to_midi(note_string), note_obj)

    def test_octave2_flat_note_generation(self):
        """ Test that Cb2 is generated correctly """

        note_string = '_Cb'  # C2b
        note_obj = note.Note('C2b')
        self.assertEqual(convert_note_to_midi(note_string), note_obj)

    def test_octave3_flat_note_generation(self):
        """ Test that Cb3 is generated correctly """

        note_string = '.Cb'  # C3b
        note_obj = note.Note('C3b')
        self.assertEqual(convert_note_to_midi(note_string), note_obj)

    def test_octave5_flat_note_generation(self):
        """ Test that Cb5 is generated correctly """

        note_string = '^Cb'  # C5b
        note_obj = note.Note('C5b')
        self.assertEqual(convert_note_to_midi(note_string), note_obj)

    def test_octave6_flat_note_generation(self):
        """ Test that Cb6 is generated correctly """

        note_string = '*Cb'  # C6b
        note_obj = note.Note('C6b')
        self.assertEqual(convert_note_to_midi(note_string), note_obj)

    def test_note_cannot_be_generated(self):
        """ Test that an exception is raised when invalid note objects are created """

        alphabet = 'HIJKLMNOPQRSTUVWXYZ'

        for not_a_note in alphabet:
            self.assertRaises(pitch.PitchException, convert_note_to_midi, not_a_note)
            self.assertRaises(pitch.PitchException, convert_note_to_midi, not_a_note + '#')
            self.assertRaises(pitch.PitchException, convert_note_to_midi, not_a_note + 'b')

    def test_invalid_note_cannot_be_generated(self):
        """ Test that an exception is raised when an invalid note is passed in """

        self.assertRaises(ValueError, convert_note_to_midi, 'C#4b')

    def test_note_is_generated_correctly_with_instrument_and_offset(self):
        """ Test that a note is generated correctly """

        note_string = 'C'
        note_obj = parse_note(note_string, instrument.Piano(), 0)
        self.assertIsInstance(note_obj, note.Note)


class TestMelodyGeneration(unittest.TestCase):
    """ Test the generation of melodies """

    prediction = generate_random_melody(10)

    def test_all_notes_are_generated(self):
        """ Test that all notes are generated """

        stream_generated = generate_midi_notes(self.prediction, _instrument=instrument.Piano(), offset=0)
        self.assertEqual(len(stream_generated), len(self.prediction))

    def test_melodies_are_converted_to_notes_successfully(self):
        """ Test that a melody is converted to note (or chord) elements successfully """

        stream_generated = generate_midi_notes(self.prediction, _instrument=instrument.Piano(), offset=0)
        for components in stream_generated:
            if isinstance(components, note.Note):
                self.assertTrue(True)
            else:
                self.assertIsInstance(components, chord.Chord)

    def test_melody_is_generated_successfully(self):
        """ Test that a melody is generated successfully """

        stream_generated = generate_midi_notes(self.prediction, _instrument=instrument.Piano(), offset=0)
        self.assertTrue(len(stream_generated) > 0)


class TestGenerationUtilities(unittest.TestCase):
    """ Test the generation utilities (i.e. music sheets and sonifications) """

    prediction = generate_random_melody(10)

    def test_generate_xml_file(self):
        """ Test that a music xml file is generated successfully """

        cookie_id = utils.generate_random_string(16)
        filepath = '../flask_source/static/assets/music-sheet-xml'
        generate_musicxml_file(self.prediction, cookie_id, output_filepath=filepath)

        self.assertTrue(os.path.isfile(filepath + '/output' + cookie_id + '.musicxml'))

    def test_song_generation(self):
        """ Test that a midi file is generated successfully """

        cookie_id = utils.generate_random_string(16)
        filepath = generate_song(self.prediction, cookie_id, output_filepath='../flask_source/static/assets/sonifications')

        self.assertTrue(os.path.isfile(filepath))


if __name__ == '__main__':
    unittest.main()
