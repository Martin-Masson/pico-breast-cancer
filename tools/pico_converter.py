from __future__ import annotations

from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy import util
import os
import random
import argparse


class PicoConvertor:
    def __init__(self, input_dir: str, output_dir: str) -> None:
        """Initializes the instance based on the input and output directory names.

        Args:
          input_dir: A string of the relative path of the directory where the
            BRAT annotated files are located.
          output_dir: A string of the relative path of the directory where the
            IOB split files will be created.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.tokenizer = PicoConvertor.make_tokenizer()

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
    
    @staticmethod
    def get_text(txt_file: str) -> str:
        """Returns the content of the input text file as a string."""
        with open(txt_file, 'r') as f:
            text = f.read()

        return text.strip()
    
    @staticmethod
    def get_annotations(ann_file: str) -> list[dict[str, str | int]]:
        """Returns the useful annotations of the input BRAT file.

        Args:
            ann_file: A string of the relative path of the BRAT annotation file.

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

    def get_file_pairs(self) -> list[tuple[str, str]]:
        """Pairs the ann and txt files of the input folder.

        Loops over all the .ann files of the input directory checking if a .txt
        file with the same name exists before adding them to a list.

        Returns:
            A list of tuples (*.ann, *.txt), where *.ann is a string of the relative
              path of the BRAT annotated file and *.txt is a string of the relative
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
                    (f"{self.input_dir}/{ann_file}",
                     f"{self.input_dir}/{txt_file}")
                )
            else:
                raise f"{ann_file} does not have a corresponding text file."

        return file_pairs

    def convert_split(self,
                      split: list[tuple[str, str]],
                      split_file: str,
                      all_file: str) -> None:
        """Convert all files of a split.
        
        Loops over all the file pairs of a split and write each token and its
        label to both the global and the split's corresponding output file.

        Args:
          split: A slice of the list returned by get_file_pairs().
          split_file: A string of the relative path of the split's output file.
          split_file: A string of the relative path of the global output file.
        """
        with open(split_file, 'w') as split_f, open(all_file, 'a') as all_f:
            for ann_file, txt_file in split:
                annotations = self.get_annotations(ann_file)
                len_annotations = len(annotations)
                tokens = self.tokenizer(self.get_text(txt_file))

                annotation_idx = 0  # The index of annotation within annotations
                tag_start = annotations[annotation_idx]["start"]
                tag_end = annotations[annotation_idx]["end"]
                tag_span = tokens.char_span(tag_start, tag_end)
                for token in tokens:
                    if token not in tag_span:
                        if not token.text.isspace():
                            split_f.write(f"{token} O\n")
                            all_f.write(f"{token} O\n")
                    else:
                        ner_tag = annotations[annotation_idx]["ner_tag"]
                        # Whether the token is at the beginning or in the middle of the tagged text
                        pos = "B" if token == tag_span[0] else "I"
                        if not token.text.isspace():
                            split_f.write(f"{token} {pos}-{ner_tag}\n")
                            all_f.write(f"{token} {pos}-{ner_tag}\n")
                        # If the token is the last of tagged text
                        if token == tag_span[-1]:
                            annotation_idx += 1
                            if annotation_idx < len_annotations:
                                tag_start = annotations[annotation_idx]["start"]
                                tag_end = annotations[annotation_idx]["end"]
                                tag_span = tokens.char_span(tag_start, tag_end)

                split_f.write("\n")
                all_f.write("\n")
        
    def convert(self) -> None:
        """Convert all the files into splits.
        
        Converts and splits the annotations into train, validation and test
        files containing 80%/10%/10% of the total annotations.
        """
        file_pairs = self.get_file_pairs()
        random.Random(42).shuffle(file_pairs)
        split_size = len(file_pairs)//10

        # Evenly distribute rare labels among all three splits
        test_pair = (f"{self.input_dir}/12621740.ann", f"{self.input_dir}/12621740.txt")
        dev_pair = (f"{self.input_dir}/16897238.ann", f"{self.input_dir}/16897238.txt")
        train_pair = (f"{self.input_dir}/31490251.ann", f"{self.input_dir}/31490251.txt")
        file_pairs.remove(test_pair)
        file_pairs.remove(dev_pair)
        file_pairs.remove(train_pair)
        file_pairs.insert(0, test_pair)
        file_pairs.insert(split_size, dev_pair)
        file_pairs.insert(2*split_size, train_pair)
        
        test_split = file_pairs[0:split_size]
        dev_split = file_pairs[split_size:2*split_size]
        train_split = file_pairs[2*split_size:]
        splits = [test_split, dev_split, train_split]

        test_file = f"{self.output_dir}/test.txt"
        dev_file = f"{self.output_dir}/dev.txt"
        train_file = f"{self.output_dir}/train.txt"
        all_file = f"{self.output_dir}/all.txt"
        split_files = [test_file, dev_file, train_file]

        for split, split_file in zip(splits, split_files):
            self.convert_split(split, split_file, all_file)


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
        "--output_dir",
        default="pico_iob",
        type=str,
        help="Output directory where IOB annotations splits are saved.",
        dest="output_dir",
    )

    args = parser.parse_args()
    pico_converter = PicoConvertor(args.input_dir, args.output_dir)
    pico_converter.convert()
