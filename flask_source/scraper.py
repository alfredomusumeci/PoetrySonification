""" The webscraper script to download the (lyric, notes) pairs """
from bs4 import BeautifulSoup
from string import punctuation
from urllib.error import URLError
import requests
import time
import pickle


def parse_notes_line(line):
    """ Process a set of notes and output them in the correct format.
    Accepted combinations are: (^, ., *, _)Note; Note# or NoteNotelowercase;
    Note-Note(-Note)*; (^, ., *, _)Note#; Note.
    The symbols (^, ., *, _) indicate the registry, while the symbol #,
    or a note represented by a lower case, represent the corresponding flat note
    :param line: a string containing a set of note.
    :return: a string containing the notes with correct whitespaces. """

    registry = ['^', '_', '.', '*']

    # Remove any accidental character that might have been kept during scraping.
    line = ''.join(ch for ch in line if ch.isalnum() or ch in registry + ['-', '#'])

    # A while loop to update len(line) in real time at each iteration.
    i = 0
    while i < len(line) - 1:
        if line[i].isalpha():
            # Check for the note registry and add a space.
            if line[i + 1] in registry:
                line = line[:i + 1] + ' ' + line[i + 1:]
            # Check if the note is followed by another uppercase note and add a space.
            elif line[i + 1].isalpha() and line[i + 1].isupper():
                line = line[:i + 1] + ' ' + line[i + 1:]
        # Check if the note is flat
        elif line[i] == '#' or (line[i].isalpha() and line[i].islower()):
            # If the note is followed by anything else except the '-' char, add a space.
            if line[i + 1] != '-':
                line = line[:i + 1] + ' ' + line[i + 1:]
        i += 1
    return line


def split_into_sentences(words_list, n):
    """ Split a collection of words in sentences of increasing lengths by n.
    :param words_list: the list of words to make sentences from.
    :param n: the increasing length factor.
    :return: a list of sentences. """

    # Build the splitting indexes based on the length of the words list and n.
    splitting_indexes = (list(range(1, len(words_list) + 1, n)))

    sentences = []
    # If the words list is not empty:
    if len(splitting_indexes) != 0:
        for index in splitting_indexes:
            if index <= len(words_list):
                sentences.append(words_list[:index])
        # If the last sentence is not complete, add the remaining words to it.
        if splitting_indexes[-1] != len(words_list):
            sentences.append(words_list)

    return sentences


def split_line(lyrics_line):
    """ Split a sentence into its composing words by:
    1. removing punctuation;
    2. making all letters lowercase;
    3. splitting the sentence into words.
    :param lyrics_line: a string containing a sentence.
    :return: a list of words. """

    lyrics_no_punctuation = ' '.join(word.strip(punctuation)
                                     for word in lyrics_line.split() if word.strip(punctuation))
    words = lyrics_no_punctuation.lower().split()

    return words


def validate_pair(notes_line, lyrics_line):
    """ Returns true if the number of words in a line
    (lyrics) match the number of notes
    :param lyrics_line: a string containing a sentence.
    :param notes_line: a string containing a set of notes.
    :return: a boolean value. """

    if notes_line is None or lyrics_line is None:
        return False
    elif len(notes_line) == 0 or len(lyrics_line) == 0:
        return False

    return len(notes_line) == len(lyrics_line)


############################
# Core Logic of the script #
############################

# Turn scraping on and off.
SCRAPING = False

if SCRAPING:
    # Count how many songs have been scraped.
    count = 0
    all_urls = []

    # Lists to store notes and lyrics.
    songs_notes = []  # notes.
    songs_lyrics = []  # lyrics.

    genres = []
    main = 'https://noobnotes.net/song-collections'  # main page with collection of genres.

    # Collect the URLs of the various genres.
    try:
        page = requests.get(main)
        soup = BeautifulSoup(page.content, 'html.parser')
        try:
            article_elements = soup.find('article')
            div_results = article_elements.find_all('div', class_='excerpt_class')
            for div_el in div_results:
                link = div_el.find('a', class_='blog-link')
                genres.append(link.get('href'))
        except AttributeError:
            print('Could not find one of the necessary tag to start the script, quitting')
            quit()
    except URLError:
        print('Unexpected problem with the main URL encountered, quitting the script')
        quit()

    # Collect the URLs of each song in each genre.
    for genre_url in genres:
        time.sleep(30)  # timesleep to prevent overscraping and website blocking.
        print('Currently parsing the following collection: {}'.format(genre_url))
        try:
            page = requests.get(genre_url)
            soup = BeautifulSoup(page.content, 'html.parser')
        except URLError:
            print('The URL {} does not correctly load a page, skipping to the next one.'.format(genre_url))
            continue

        # Find the tag containing the URLs to the various songs of the current genre.
        urls = []
        try:
            article_elements = soup.find('article')
            div_results = article_elements.find_all('div', class_='excerpt_class')
            for div_el in div_results:
                link_elements = div_el.find_all('a', class_='blog-link')
                for link in link_elements:
                    single_url = link.get('href')
                    if single_url not in all_urls:
                        urls.append(single_url)
                        all_urls.append(single_url)
        except AttributeError:
            print('Could not find the article tag for the following URL: {}, thus skipping to the next one.'
                  .format(genre_url))
            continue

        # Obtain the list of notes and lyrics for each song.
        for song_url in urls:
            print('Currently parsing the following song: {}'.format(song_url))
            try:
                song_page = requests.get(song_url)
                song_soup = BeautifulSoup(song_page.content, 'html.parser')

                first_line = song_soup.text.strip().partition('\n')[0]
                if first_line == 'Database Error':  # Catching error manually as it isn't handled by any library.
                    print('The URL {} shows a Database Error, thus moving to the next one.'.format(song_url))
                    continue
            except URLError:
                print('The URL {} does not correctly load a page, thus skipping to the next one.'.format(genre_url))
                continue

            try:
                # Find the div class where the notes and lyrics are stored.
                song_post = song_soup.find('article')
                div_class = song_post.select_one('div.post-content')
                p_elements_unparsed = div_class.find_all('p')
            except AttributeError:
                print('The lyrics for the song at {} could not be parsed, thus skipping to the next one'.
                      format(song_url))
                continue

            # notes and lyrics for each song.
            notes = []
            lyrics = []

            # Traverse the 'p' tags of notes and lyrics and process both of them
            # in the desired format.
            # N.B.: 'p' tags contain both notes and lyrics.
            for p_tag in p_elements_unparsed:
                # Obtains a line of lyrics.
                br_element = p_tag.find('br')

                # Scrape the corresponding note for the given lyrics only if these
                # exist or if the text is not empty; this is due to some irregularities
                # with the HTML code used in the website.
                if br_element is not None and br_element.get_text() != '':
                    parsed_br = br_element.get_text().replace(u'\u200b', '').replace(u'\xa0', ' ').strip()  # lyrics.
                    br_element.extract()
                    notes_no_whitespaces = p_tag.get_text().replace(u'\xa0', '').replace(' ', '')  # no whitespaces.
                    parsed_p = parse_notes_line(notes_no_whitespaces)

                    single_notes = parsed_p.split()
                    single_words = split_line(parsed_br)

                    # If there is a one-to-one correspondence between notes and words, add them to the parsed elements.
                    if validate_pair(single_notes, single_words):
                        lyrics.extend(single_words)
                        notes.extend(single_notes)

            songs_notes.extend(split_into_sentences(notes, 2))
            songs_lyrics.extend(split_into_sentences(lyrics, 2))
            count += 1

    print('Total number of songs parsed successfully: {} out of {}'.format(count, len(all_urls)))

    # Store lyrics and notes on disk for later retrieval.
    with open('flask_source/static/assets/pickle-dsts/songs_notes_split', 'wb') as notes_path:
        pickle.dump(songs_notes, notes_path)
    notes_path.close()

    with open('flask_source/static/assets/pickle-dsts/songs_lyrics_split', 'wb') as lyrics_path:
        pickle.dump(songs_lyrics, lyrics_path)
    lyrics_path.close()
