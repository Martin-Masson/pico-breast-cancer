from __future__ import annotations

from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy import util
import os
import argparse


class FormatConvertor:
    def __init__(self, input_dir: str, output_file: str) -> None:
        """Initializes the instance based on the input directory and output file names.

        Args:
          input_dir: The relative path of the directory where the BRAT
            annotated files are located.
          output_file: The relative path of the output CoNLL file.
        """
        self.input_dir = input_dir
        self.output_file = output_file

    def get_file_pairs(self) -> list[tuple[str, str]]:
        """Pairs the ann and txt files of the input folder.

        Returns:
            A list of tuples (*.ann, *.txt), where *.ann is the relative
              path of the BRAT annotated file and *.txt is the relative
              path of its corresponding text file.
        """
        file_pairs = []
        files = os.listdir(self.input_dir)
        annotation_files = sorted([file for file in files if file.endswith(".ann")])
        # The folder is assumed to contain *.ann and *.txt where both files
        # of a pair have the same name
        for ann_file in annotation_files:
            txt_file = ann_file.replace(".ann", ".txt")
            if txt_file in files:
                file_pairs.append(
                    (os.path.join(self.input_dir, ann_file),
                     os.path.join(self.input_dir, txt_file))
                )
            else:
                raise f"{ann_file} does not have a corresponding text file."

        return file_pairs

    @staticmethod
    def get_annotations(ann_file: str) -> list[dict[str, str | int]]:
        """Reads the useful annotations of the input BRAT file.

        Args:
            ann_file: The relative path of the BRAT annotated file.

        Returns:
            A list of dicts each mapping a line's tag, start and end to
            their corresponding value.
        """
        annotations = []
        # Read each line of the annotation file to a dictionary
        with open(ann_file, 'r') as f:
            for line in f:
                if line != "\n":
                    line_annotations = {}
                    splits = line.split()
                    line_annotations["ner_tag"] = splits[1].lower()
                    line_annotations["start"] = int(splits[2])
                    line_annotations["end"] = int(splits[3])
                    annotations.append(line_annotations)
        # Sorts the annotations by start position to be able to loop through
        # the text while iterating over the list
        annotations.sort(key=lambda x: x["start"])
        return annotations

    @staticmethod
    def get_text(txt_file: str) -> str:
        """Returns the content of the input text file as a string."""
        with open(txt_file, 'r') as f:
            text = f.read()

        return text

    @staticmethod
    def make_tokenizer() -> Tokenizer:
        """Returns a spacy tokenizer with special rule changes for the task."""
        nlp = English()
        tokenizer = nlp.tokenizer

        prefixes = nlp.Defaults.prefixes + [r"(\d+(\.\d+)?)"]
        suffixes = nlp.Defaults.suffixes + [r"(\d+(\.\d+)?)", r"\."]
        infixes = nlp.Defaults.infixes + [r"(\d+(\.\d+)?)", r"--", r"\(", r"\)", r"/"]

        prefix_regex = util.compile_prefix_regex(prefixes)
        suffix_regex = util.compile_suffix_regex(suffixes)
        infix_regex = util.compile_infix_regex(infixes)

        tokenizer.prefix_search = prefix_regex.search
        tokenizer.suffix_search = suffix_regex.search
        tokenizer.infix_finditer = infix_regex.finditer

        rules = tokenizer.rules.items()
        tokenizer.rules = {k: v for k, v in rules if k not in ['):', '8)']}

        return tokenizer

    def convert(self) -> None:
        """Loops over all the file pairs and write each token and its label to the output file."""
        tokenizer = self.make_tokenizer()

        with open(self.output_file, 'w') as f:
            for ann_file, txt_file in self.get_file_pairs():
                annotations = self.get_annotations(ann_file)
                tokens = tokenizer(self.get_text(txt_file))
                len_annotations = len(annotations)

                annotation_idx = 0  # The index of annotation within annotations
                tag_start = annotations[annotation_idx]["start"]
                tag_end = annotations[annotation_idx]["end"]
                tag_span = tokens.char_span(tag_start, tag_end)
                for token in tokens:
                    if token not in tag_span:
                        f.write(f"{token} O\n")
                    else:
                        ner_tag = annotations[annotation_idx]["ner_tag"]
                        # Whether the token is at the beginning or in the middle of the tagged text
                        pos = "B" if token == tag_span[0] else "I"
                        f.write(f"{token} {pos}-{ner_tag}\n")
                        # If the token is the last of tagged text
                        if token == tag_span[-1]:
                            annotation_idx += 1
                            if annotation_idx < len_annotations:
                                tag_start = annotations[annotation_idx]["start"]
                                tag_end = annotations[annotation_idx]["end"]
                                tag_span = tokens.char_span(tag_start, tag_end)

                f.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input_dir",
        default="brat_annotations",
        type=str,
        help="Input directory where Brat annotations are located.",
        dest="input_dir"
    )

    parser.add_argument(
        "-o",
        "--output_file",
        default="pico_conll.txt",
        type=str,
        help="Output file where CoNLL annotations are saved.",
        dest="output_file",
    )

    args = parser.parse_args()
    format_convertor = FormatConvertor(args.input_dir, args.output_file)
    format_convertor.convert()
