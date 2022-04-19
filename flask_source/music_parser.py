""" The parser utilised to transform sonifications into actual music (.mid files). """
from music21 import note, instrument, chord, stream, environment
import music21.musicxml.m21ToXml


def generate_midi_notes(prediction, _instrument, offset):
    """ Generate a list of notes and chords from their string.
    :param prediction: a string containing a prediction made of notes and chords.
    :param _instrument: a music21 compatible instrument object.
    :param offset: a pre-established offset to avoid stacking of notes.
    :return: a list with notes and chords objects. """

    _stream = []

    for pattern in prediction:
        # If pattern is a chord.
        if '-' in pattern:
            chord_components = pattern.split('-')

            notes = []
            for current_note in chord_components:
                notes.append(parse_note(current_note, _instrument, offset))

            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            _stream.append(new_chord)

        # If pattern is a note.
        else:
            note_obj = parse_note(pattern, _instrument, offset)
            _stream.append(note_obj)

        # Avoid stacking of notes.
        offset += 0.85

    return _stream


def pick_octave(symbol):
    """ Pick the corresponding octave number.
    :param symbol: a string representing the octave.
    :return: a number representing the octave. """

    if symbol == '_':
        return 2
    elif symbol == '.':
        return 3
    elif symbol == '^':
        return 5
    else:  # symbol == '*'
        return 6


def convert_note_to_midi(prediction):
    """ Create the corresponding note object from a prediction string.
    :param prediction: a string representing a single note.
    :return: a note object. """

    # A note can only be made of a letter, an octave and a sharp/flat symbol.
    if len(prediction) > 3:
        raise ValueError('The note passed is invalid.')

    parsed_note = ''
    # If the note has standard octave.
    if prediction[0].isalpha():
        parsed_note += prediction[0] + '4'
    # If the note has a custom octave.
    else:
        parsed_note += prediction[1] + str(pick_octave(prediction[0]))

    # If the note is flat.
    if prediction[-1] == '#' or (prediction[-1].isalpha() and prediction[-1].islower()):
        parsed_note += prediction[-1]

    return note.Note(parsed_note)


def parse_note(_note, _instrument, offset):
    """ Add an instrument and an offset to a note object.
    :param _note: a string representing a single note.
    :param _instrument: a music21 compatible instrument object.
    :param offset: a pre-established offset to avoid stacking of notes.
    :return: a note object with personalised offset and instrument. """

    parsed_note = convert_note_to_midi(_note)
    parsed_note.offset = offset
    parsed_note.storedInstrument = _instrument

    return parsed_note


def create_parser_from_stream(midi_stream, encoding='utf-8'):
    """ Create a music21 parser from a generated stream.
    :param midi_stream: a Stream object.
    :param encoding: a compatible encoding.
    :return: a parser object using the provided encoding. """

    GEX = music21.musicxml.m21ToXml.GeneralObjectExporter(midi_stream)
    out = GEX.parse()
    return out.decode(encoding)


def generate_musicxml_file(prediction, cookie_id, _instrument=instrument.Piano(), offset=0,
                           output_filepath='flask_source/static/assets/music-sheet-xml'):
    """ Generate a musicxml file for the given prediction. The file produced is saved
    inside static/assets/music-sheet-xml.
    :param prediction: a string containing a prediction made of notes and chords.
    :param cookie_id: a string representing the cookie id.
    :param _instrument: a music21 compatible instrument object.
    :param offset: a pre-established offset to avoid stacking of notes.
    :param output_filepath: a string representing the output filepath. """

    output_stream = generate_midi_notes(prediction, _instrument, offset)
    midi_stream = stream.Stream(output_stream)

    out_string = create_parser_from_stream(midi_stream)

    with open(output_filepath + '/output{id}.musicxml'.format(id=cookie_id), "w") as music_xml:
        music_xml.write(out_string.strip())


def generate_music_sheet_png(prediction, _instrument, offset, musescore_path):
    """ Generate a music sheet png for the given prediction. This requires
    MuseScore to be installed. The file produced is immediately shown.
    :param prediction: a string containing a prediction made of notes and chords.
    :param _instrument: a music21 compatible instrument object.
    :param offset: a pre-established offset to avoid stacking of notes.
    :param musescore_path: the path to the MuseScore binary files. """

    output_stream = generate_midi_notes(prediction, _instrument, offset)
    midi_stream = stream.Stream(output_stream)

    us = environment.UserSettings()
    us['musescoreDirectPNGPath'] = musescore_path

    midi_stream.show('musicxml.png')


def generate_song(prediction, cookie_id, _instrument=instrument.Piano(), offset=0,
                  output_filepath='flask_source/static/assets/sonifications'):
    """ Generate a midi file from a given prediction and instrument.
    The file is saved at the indicated destination folder.
    :param prediction: a string containing a prediction made of notes and chords.
    :param cookie_id: a string representing the cookie id.
    :param _instrument: a music21 compatible instrument object.
    :param offset: a pre-established offset to avoid stacking of notes.
    :param output_filepath: the path to the destination folder.
    :return: the path to the generated midi file. """

    output_stream = generate_midi_notes(prediction, _instrument, offset)
    midi_stream = stream.Stream(output_stream)

    midi_filepath = midi_stream.write('midi', fp=output_filepath + '/output{id}.mid'.
                                      format(id=cookie_id))

    # Uncomment this to play the file immediately
    # Only works if a media player that supports .mid files is installed in the host machine.
    # midi_stream.show('midi')

    return midi_filepath


