""" DEPRECATED. Used to write dataset to a csv file, now using pickle. """
import pickle
import csv

# Restore lyrics and notes data structures.
# N.B.: path must be changed if used.
with open('tmp/data_structures/songs_lyrics', 'rb') as lyrics_path:
    lyrics = pickle.load(lyrics_path)
lyrics_path.close()

with open('tmp/data_structures/songs_notes', 'rb') as notes_path:
    notes = pickle.load(notes_path)
notes_path.close()

song_notes = list(zip(lyrics, notes))
word_notes = []
for song_notes_pair in song_notes:
    song_lyrics = song_notes_pair[0]
    song_notes = song_notes_pair[1]
    word_notes.append(list(zip(song_lyrics, song_notes)))

# Write (word, notes) pairs in a csv file to be used for
# training of the model later.
with open('flask_source/resources/word-note-pairs.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for song in word_notes:
        writer.writerows(song)
        writer.writerow([])

file.close()

print('Writing into csv file completed')
print('All songs and lyrics have been parsed successfully: {}'.format(len(notes) == len(lyrics) == len(word_notes)))
