from typing import List, Dict
import html
import os
from sklearn.metrics import  mean_absolute_percentage_error
import re
import numpy as np
from tqdm import tqdm
import argparse
from scipy.optimize import curve_fit

POWER = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900,
         1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000]


def preprocess_text(text):
    """
    Convert all named and numeric character references (e.g. &gt;, &#62;,
    &x3e;) in the string s to the corresponding unicode characters.
    This function uses the rules defined by the HTML 5 standard
    for both valid and invalid character references, and the list of
    HTML 5 named character references defined in html.entities.html5.
    Args:
        text: Book text.

    Returns:
        Book text in lowercase letters.
    """
    text = html.unescape(text)  # &amp;#x200B; => &#x200B;
    text = html.unescape(text)  # &amp;#x200B; =>
    return text.lower()


def tokenize(text):
    """
    Return the string obtained by replacing the leftmost
    non-overlapping occurrences of the pattern in string by the
    replacement repl.  repl can be either a string or a callable;
    if a string, backslash escapes in it are processed.  If it is
    a callable, it's passed the Match object and must return
    a replacement string to be used.
    Args:
        text: Book text.

    Returns:
        List of words.
    """
    opt = re.sub(r'[^\w\s]', '', text)
    words = opt.split()
    return words


def get_embeddings(path_glove: str) -> Dict[str, np.ndarray]:
    """
    The function returns a dictionary, the key is a word, the value is a vector embedding
    Args:
        path_glove: Path to the glove file.

    Returns:
        The function returns a dictionary Dict[str, np.ndarray], the key is a word, the value is a vector embedding
    """
    embeddings = {}
    with open(path_glove, 'r', encoding='utf-8') as file:

        for line in tqdm(file):
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings[word] = embedding
    return embeddings


def get_tokens(path_text: str) -> List[str]:
    """
    The function returns a list of tokens from text.
    Args:
        path_text: Path to the text file.

    Returns:
        List of tokens from text.
    """
    tokens = []
    with open(path_text, encoding='utf-8') as file:
        for line in file:
            tokens += tokenize(preprocess_text(line))
    return tokens


def get_tokens_with_embeddings(embeddings: np.ndarray, tokens: List, verbose: bool = False) -> np.ndarray:
    """
    The function finds embeddings of words present in the text.
    Args:
        embeddings: Embeddings.
        tokens: Tokens.
        verbose: if true, output to console words that arevnot in the GloVe dictionary 
    Returns:
        Embeddings of words that are in the book.
    """
    tokens_with_embeddings = []
    for token in tokens:
        try:
            tokens_with_embeddings.append(embeddings[token])
        except KeyError:
            if verbose:
                print('Not found', token)

    return np.asarray(tokens_with_embeddings, dtype='float32')


def get_normalized_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    The function finds the weighted average values of embeddings.
    Args:
        embeddings: Embeddings.

    Returns:
        The function returns array weighted average values of embeddings.
    """
    number, dim = embeddings.shape
    avg = np.zeros(dim)
    for i in range(number):
        avg += embeddings[i]
    avg = avg / number

    for i in range(number):
        embeddings[i] -= avg

    return embeddings


def get_norms_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    The function finds the norm of vectors.
    Args:
        embeddings: Embeddings.

    Returns:
        Norm of word vectors.
    """
    number = len(embeddings)
    norms = np.zeros(number)

    for i in range(number):
        norms[i] = np.linalg.norm(embeddings[i])
    return norms


def get_autocorrelation(embeddings: np.ndarray, norms: np.ndarray, number_embeddings: int) -> np.ndarray:
    """
    The function finds the autocorrelation values of word vectors
    Args:
        embeddings: Embeddings.
        norms: Norm of word vectors.
        number_embeddings: Number of Embeddings.

    Returns:
        Array of correlation values
    """
    autocorr = np.zeros(len(POWER))
    for k in tqdm(range(len(POWER))):
        sumcorr = 0.
        j = POWER[k]
        for i in range(number_embeddings - j):
            corr = np.dot(embeddings[i], embeddings[i + j]) / norms[i] / norms[i + j]
            sumcorr += corr
        autocorr[k] = sumcorr / (number_embeddings - j)
    return autocorr


class GAPELMAPER:
    """
    Implements GAPELMAPER metric.

    Args:
        self.values_x:
        self.values_y: Array of correlation values
        self.negative_index: Index of the first non-negative value after the last index of the negative value.

    """

    def __init__(self, values_x: np.array, values_y: np.array):
        self.negative_index = 0
        self.values_x = values_x
        self.values_y = values_y
        self.y_fitpow = None
        self.y_fitexp = None
        self.y_fitlog = None
        self.__get_negative_index()

    def __get_negative_index(self):
        """The method finds the extreme (maximum) index of a non-negative value"""
        neg = np.where(self.values_y <= 0)[0]
        if neg.size > 0:
            self.negative_index = np.max(neg) + 1

    @staticmethod
    def __mapping(x: np.array, a: float, b: float) -> np.array:
        """
        The method approximates the direct correlation value.

        Args:
            x: the coordinates of any point on the line
            a: is the slope of the line.
            b: is the y-intercept (where the line crosses the y-axis).

        Return:
            Returns an array of numbers.
        """
        return a * x + b

    def get_power_mape(self):
        """
            The method finds the average absolute percentage error for a power autocorrelation decay law.
        Returns:
            The method returns the average absolute percentage error for a power autocorrelation decay law.
        """
        values_x = np.log10(self.values_x[self.negative_index:])
        values_y = np.log10(self.values_y[self.negative_index:])

        argspow, _ = curve_fit(self.__mapping, values_x, values_y)
        self.y_fitpow = self.__mapping(values_x, argspow[0], argspow[1])
        return mean_absolute_percentage_error(self.values_y[self.negative_index:], 10 ** self.y_fitpow)

    def get_log_mape(self):
        """
            The method finds the average absolute percentage error for a logarithmic autocorrelation decay law.
        Returns:
            The method returns the average absolute percentage error for a logarithmic autocorrelation decay law.
        """
        values_x = np.log10(self.values_x[self.negative_index:])
        values_y = self.values_y[self.negative_index:]

        argslog, _ = curve_fit(self.__mapping, values_x, values_y)
        self.y_fitlog = self.__mapping(values_x, argslog[0], argslog[1])
        return mean_absolute_percentage_error(values_y, self.y_fitlog)

    def get_exp_mape(self):
        """
            The method finds the average absolute percentage error for an exponential autocorrelation decay law.
        Returns:
            The method returns the average absolute percentage error for an exponential autocorrelation decay law.
        """
        values_x = self.values_x[self.negative_index:]
        values_y = np.log10(self.values_y[self.negative_index:])

        argsexp, _ = curve_fit(self.__mapping, values_x, values_y)
        self.y_fitexp = self.__mapping(values_x, argsexp[0], argsexp[1])
        return mean_absolute_percentage_error(self.values_y[self.negative_index:], 10 ** self.y_fitexp)

    @staticmethod
    def get_gapelmaper(power_mape, exp_mape):
        """
        Find GAPELMAPER.
        Args:
            power_mape: Average absolute percentage error for a power law.
            exp_mape: Average absolute percentage error for an exponential law

        Returns:
            Metric GAPELMAPER.
        """
        return power_mape / exp_mape


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Calculate GAPELMAPER')
    parser.add_argument('path_glove',
                        type=str,
                        help='The path to the file glove')
    parser.add_argument('path_text',
                        type=str,
                        help='The path to the file text')

    args = parser.parse_args()
    _, filename = os.path.split(args.path_text)

    embeddings = get_embeddings(args.path_glove)
    print('Found %s word vectors.' % len(embeddings))

    tokens = get_tokens(args.path_text)
    print('All tokens in', filename, len(tokens))

    tokens_with_embeddings = get_tokens_with_embeddings(embeddings, tokens, verbose=False)
    print("tokens with embeddings", len(tokens_with_embeddings))

    normalized_embeddings = get_normalized_embeddings(tokens_with_embeddings)
    print('Embeddings normalized', normalized_embeddings.shape, normalized_embeddings.mean())

    quantity_embeddings = len(normalized_embeddings)
    norms = get_norms_embeddings(normalized_embeddings)
    autocorr = get_autocorrelation(normalized_embeddings, norms, quantity_embeddings)
    print(filename)

    metric = GAPELMAPER(np.array(POWER), autocorr)

    print('Power decay')
    MAPE_POWER = metric.get_power_mape()
    print('MAPE =', MAPE_POWER)
    print()
    print('Log decay')
    MAPE_LOG = metric.get_log_mape()
    print('MAPE =', MAPE_LOG)
    print()
    print('Exp decay')
    MAPE_EXP = metric.get_exp_mape()
    print('MAPE =', MAPE_EXP)
    print()
    print('GAPELMAPER')
    gapelmaper = metric.get_gapelmaper(MAPE_POWER, MAPE_EXP)
    print(gapelmaper)

    with open(f'Result_GAPELMAPER_{filename}', 'w') as f:
        f.write(f'{filename}\n'
                f'Power Mape: {MAPE_POWER}\n'
                f'Log Mape: {MAPE_LOG}\n'
                f'Exp Mape: {MAPE_EXP}\n'
                f'GAPELMAPER: {gapelmaper}')
