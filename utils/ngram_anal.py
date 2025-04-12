import pandas as pd

def find_ngrams(input_list, n):
    """
    Generate n-grams from the input list.

    :param input_list: List of elements (e.g., characters)
    :param n: The number of elements in each n-gram
    :return: A generator of n-grams as tuples
    """
    return zip(*[input_list[i:] for i in range(n)])

def get_ngrams_dataframe(strings, n):
    """
    Create a DataFrame where each row is an n-gram from each string.

    :param strings: List of strings to analyze.
    :param n: The size of each n-gram (e.g., 3 for trigrams).
    :return: A pandas DataFrame with columns for the original string and the n-gram.
    """
    rows = []
    for s in strings:
        # Convert the string into a list of characters
        characters = list(s)
        # Generate n-grams for the string
        ngrams = list(find_ngrams(characters, n))
        # Append each n-gram as a new row in the list with the original string
        for ngram in ngrams:
            rows.append({'string': s, 'ngram': ''.join(ngram)})
    return pd.DataFrame(rows)

# Example usage:
strings_to_analyze = passwords
n = 3  # For trigrams
df = get_ngrams_dataframe(strings_to_analyze, n)
# print(df)
ngram_df = df.groupby('ngram').agg(cnt_pass=('string','nunique'))