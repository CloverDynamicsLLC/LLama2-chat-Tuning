import ast
import os

import datasets
import torch
from torch.utils.data import Dataset

_CITATION = """\
  title={The movie dialogs for training},
  booktitle={},
  year={2024}
}
"""

_DESCRIPTION = """\
This corpus contains a large metadata-rich collection of fictional conversations extracted from raw movie scripts:
- 220,579 conversational exchanges between 10,292 pairs of movie characters
- involves 9,035 characters from 617 movies
- in total 304,713 utterances
- movie metadata included:
    - genres
    - release year
    - IMDB rating
    - number of IMDB votes
    - IMDB rating
- character metadata included:
    - gender (for 3,774 characters)
    - position on movie credits (3,321 characters)
"""

_URL = "https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"


class CornellMovieDialog(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.1.0")

    SPLITER = "+++$+++"

    def __init__(self, **kwargs):
        self.validation_size = 0
        self.train_size = 0

        if "train_size" in kwargs:
            if type(kwargs.get("train_size")) is int:
                self.train_size = kwargs.get("train_size")
            else:
                raise TypeError("train_size should be int")

        if "validation_size" in kwargs:
            if type(kwargs.get("validation_size")) is int:
                self.train_size = -1 * kwargs.get("validation_size")
            else:
                raise TypeError("validation_size should be int")

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "movieID": datasets.Value("string"),
                    "movieTitle": datasets.Value("string"),
                    "movieYear": datasets.Value("string"),
                    "movieIMDBRating": datasets.Value("string"),
                    "movieNoIMDBVotes": datasets.Value("string"),
                    "movieGenres": datasets.features.Sequence(datasets.Value("string")),
                    "characterID1": datasets.Value("string"),
                    "characterID2": datasets.Value("string"),
                    "characterName1": datasets.Value("string"),
                    "characterName2": datasets.Value("string"),
                    "utterance": datasets.features.Sequence(
                        {"text": datasets.Value("string"), "LineID": datasets.Value("string")}
                    ),
                    "dialog": datasets.features.Sequence(datasets.Value("string"))
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        dl_dir = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepaths": os.path.join(dl_dir, "cornell movie-dialogs corpus"), "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepaths": os.path.join(dl_dir, "cornell movie-dialogs corpus"), "split": "validation"},
            )
        ]

    def _generate_examples(self, filepaths, split):
        """Yields examples."""
        movie_char_file = os.path.join(filepaths, "movie_characters_metadata.txt")
        movie_conv_file = os.path.join(filepaths, "movie_conversations.txt")
        movie_lines_file = os.path.join(filepaths, "movie_lines.txt")
        movie_titles_file = os.path.join(filepaths, "movie_titles_metadata.txt")

        with open(movie_char_file, "rb") as f:
            movie_char_data = [x.decode("latin").split(self.SPLITER) for x in f.readlines()]

        if split == "train":
            with open(movie_conv_file, "rb") as f:
                if self.train_size > 0:
                    movie_conv_data = [x.decode("latin").split(self.SPLITER) for x in f.readlines()][:self.train_size]
                else:
                    movie_conv_data = [x.decode("latin").split(self.SPLITER) for x in f.readlines()]
        else:
            with open(movie_conv_file, "rb") as f:
                if self.validation_size > 0:
                    movie_conv_data = [x.decode("latin").split(self.SPLITER) for x in f.readlines()][
                                      self.validation_size:]
                else:
                    movie_conv_data = [x.decode("latin").split(self.SPLITER) for x in f.readlines()]

        with open(movie_lines_file, "rb") as f:
            movie_lines_data = [x.decode("latin").split(self.SPLITER) for x in f.readlines()]

        with open(movie_titles_file, "rb") as f:
            movie_titles_data = [x.decode("latin").split(self.SPLITER) for x in f.readlines()]
        # looping over movie conversation file
        for id_, conv in enumerate(movie_conv_data):
            char_id_1 = conv[0]
            char_id_2 = conv[1]
            movie_id = conv[2]
            line_ids = conv[-1].replace("\n", "")
            line_ids = ast.literal_eval(line_ids.strip())
            lines_texts = []
            dialog = []
            # searching text corresponding to each lineID in line_ids in movie lines file
            for line_id in line_ids:
                i = 0
                while i < len(movie_lines_data) and movie_lines_data[i][0].strip() != line_id:
                    i += 1
                line = movie_lines_data[i][4]
                lines_texts.append(line)  # if i < len(movie_lines_data) else '')
                dialog.append(f"[{movie_lines_data[i][3]}]: {line}")
            # look for char names in movie character file
            j = 0
            while j < len(movie_char_data) and movie_char_data[j][0].strip() != char_id_1.strip():
                j += 1
            char_name_1 = movie_char_data[j][1]  # if j < len(movie_char_data) else ''
            movie_title = movie_char_data[j][3]  # if j < len(movie_char_data) else ''

            k = 0
            while k < len(movie_char_data) and movie_char_data[k][0].strip() != char_id_2.strip():
                k += 1
            char_name_2 = movie_char_data[k][1]

            # look for movie year, IMDBRating, genre, no_imdb_voting in movie tiles file
            li = 0
            while li < len(movie_titles_data) and movie_titles_data[li][0].strip() != movie_id.strip():
                li += 1
            movie_year = movie_titles_data[li][2]
            imdb_rating = movie_titles_data[li][3]
            no_imdb_vote = movie_titles_data[li][4]
            genre = movie_titles_data[li][5].replace("\n", "").strip()
            movie_genres = ast.literal_eval(genre)

            yield id_, {
                "movieID": movie_id,
                "movieTitle": movie_title,
                "movieYear": movie_year,
                "movieIMDBRating": imdb_rating,
                "movieNoIMDBVotes": no_imdb_vote,
                "movieGenres": movie_genres,
                "characterID1": char_id_1,
                "characterID2": char_id_2,
                "characterName1": char_name_1,
                "characterName2": char_name_2,
                "utterance": {"text": lines_texts, "LineID": line_ids},
                "dialog": dialog
            }


class DialogDataset(Dataset):
    def __init__(self, dataset, tokenizer, block_size=512):
        self.tokenizer = tokenizer
        self.block_size = block_size

        # Load and process the dialogs
        self.dialogs = []
        for data in dataset:
            dialog = data['dialog']
            self.dialogs.append(self.process_dialog('\n'.join(dialog)))

    def process_dialog(self, dialog):
        # Tokenize each dialog
        return self.tokenizer.encode(dialog)[:self.block_size]

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return torch.tensor(self.dialogs[idx], dtype=torch.long)


def generate_dialog_response(model, tokenizer, prompt, max_length=50):
    # Encode the prompt text
    encoded_input = tokenizer.encode(prompt, return_tensors='pt')
    encoded_input = encoded_input.to('cpu')  # Move to the same device as your model

    # Generate a response
    output = model.generate(
        encoded_input,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    # Decode the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response
