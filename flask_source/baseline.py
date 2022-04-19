""" Baseline model: takes a piece of text and generates notes randomly for each word. """
import random

notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
octave = ['_', '.', '^', '*']


def add_octave_randomly(note):
    """ Decide, based on a coin toss, whether to add an octave
     to a note string.
    :param note: a string containing a note.
    :return: a string with the same note or using a different octave. """

    # Toss a coin, if 1 add the octave.
    if random.choice([0, 1]):
        return random.choice(octave) + note

    return note


def generate_random_melody(length):
    """ Generate a random melody based on the length and words of
     the inputted text.
    :param length: the desired length of the melody.
    :return: a string with a random melody drawn from the inputted text. """

    random_melody = []

    # Iterate until number of notes/chords matches number of words.
    while len(random_melody) < length:
        # Toss a coin: 0 for note, 1 for chord:
        result = random.choice([0, 1])
        if not result:
            random_melody.append(add_octave_randomly(random.choice(notes)))
        else:
            chord = add_octave_randomly(random.choice(notes))
            # Toss a coin: if 1 make the chord longer.
            while random.choice([0, 1]):
                chord += '-' + add_octave_randomly(random.choice(notes))
            random_melody.append(chord)

    return random_melody
