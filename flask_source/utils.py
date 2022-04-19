""" An utility class where various functions are stored.
These do not fit in any other class, according to the principle of encapsulation. """
import random
import os

hexadecimal_string = '0123456789abcdef'


def generate_random_string(length):
    """ Generate a random string of given length.
    :param length: The length of the string to be generated.
    :return: A random string of given length. """

    return ''.join(random.choice(hexadecimal_string) for _ in range(length))


def delete_file(file_path):
    """ Delete a file.
    :param file_path: The path of the file to be deleted. """

    try:
        os.remove(file_path)
    except OSError:
        print("Cannot delete: no such file.")
