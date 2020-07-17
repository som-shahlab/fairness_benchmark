import random


def random_string(num_characters=5):
    """
    Generates a random string of length N.
    See https://pythonexamples.org/python-generate-random-string-of-specific-length/
    """
    return "".join(
        random.choice(string.ascii_lowercase + string.digits)
        for _ in range(num_characters)
    )
