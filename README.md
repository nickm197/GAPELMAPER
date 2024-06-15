# GAPELMAPER
Reference implementation of GAPELMAPER metric from [Long Story Generation Challenge](https://aclanthology.org/2023.inlg-genchal.2.pdf)

GAPELMAPER (GloVe Autocorrelations Power/Exponential Law Mean Absolute Percentage Error Ratio) is a metric designed to assess text coherence based on the autocorrelation of embeddings. It helps determine whether the text is intrinsically structured or not, based on the decay patterns of the autocorrelations.


The first step in calculating GAPELMAPER is preprocessing the text, which includes cleaning, tokenization, and creating an embedding dictionary. To begin, we define the preprocess_text function to transform HTML  entities into corresponding characters and convert the entire text to lowercase. Then, using another function tokenize, we remove all characters (except letters and spaces) and split our text into some tokens.

Next, we create a \[‘word': emb\]-like dictionary using a pre-existing list of GloVe model embeddings. Utilising the get_tokens_with_embeddings function, we obtain the embeddings we are interested in (of those words that are contained in the text). The next step is to calculate the norms of the embedding vectors. For this purpose, we define the get_norms_embeddings function. Then, we compute the autocorrelation for the embeddings using the dot product of vectors for various shifts.

Finally, we initialize the GAPELMAPER class. It receives arrays of values values_x and values_y, representing the values for the x and y axes. After this, we calculate the MAPE (mean absolute percentage error). Speaking specifically, get_power_mape calculates MAPE for the power law, get_log_mape calculates MAPE for the logarithmic law, and get_exp_mape calculates MAPE for the exponential law.
Then, the get_gapelmaper method takes the MAPE values for both power and exponential laws and returns their ratio, which is the GAPELMAPER metric. If GAPELMAPER < 1, the autocorrelations decay according to the power law, indicating that the text is structured. Conversely, if GAPELMAPER > 1, the autocorrelations decay according to the exponential law, indicating that the text is unstructured.

If you use this code  in your research, please cite as appropriate:

```
@inproceedings{mikhaylovskiy-2023-long,
    title = "Long Story Generation Challenge",
    author = "Mikhaylovskiy, Nikolay",
    editor = "Mille, Simon",
    booktitle = "Proceedings of the 16th International Natural Language Generation Conference: Generation Challenges",
    month = sep,
    year = "2023",
    address = "Prague, Czechia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.inlg-genchal.2",
    pages = "10--16",
}

@inproceedings{mikhaylovskiy2023autocorrelations,
  title={Autocorrelations Decay in Texts and Applicability Limits of Language Models},
  author={Mikhaylovskiy, Nikolay and Churilov, Ilya},
  booktitle={Computational Linguistics and Intellectual Technologies: Proceedings of the International Conference “Dialogue 2023”},
  volume={2023},
  year={2023}
}
