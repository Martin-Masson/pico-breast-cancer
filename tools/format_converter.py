from typing import Dict, List, Tuple, TypedDict

from spacy.lang.en import English
from spacy.util import compile_prefix_regex, compile_suffix_regex, compile_infix_regex
from spacy.symbols import ORTH
import os
import argparse

from pprint import pprint


class FormatConvertor:
    def __init__(self, input_dir: str, output_file: str) -> None:
        self.input_dir = input_dir
        self.output_file = output_file

    def get_file_pairs(self) -> List[Tuple[str, str]]:
        file_pairs = []
        files = os.listdir(self.input_dir)
        annotation_files = sorted([file for file in files if file.endswith(".ann")])

        for ann_file in annotation_files:
            txt_file = ann_file.replace(".ann", ".txt")
            if txt_file in files:
                file_pairs.append(
                    (os.path.join(self.input_dir, ann_file), os.path.join(self.input_dir, txt_file))
                )
            else:
                raise (f"{ann_file} does not have a corresponding text file.")
        
        return file_pairs

    def get_annotations(self, ann_file: str) -> List[TypedDict]:
        annotations = []

        with open(ann_file, 'r') as f:
            for line in f:
                if line != "\n":
                    file_annotations = {}
                    splits = line.split()
                    file_annotations["ner_tag"] = splits[1].lower()
                    file_annotations["start"] = int(splits[2])
                    file_annotations["end"] = int(splits[3])
                    file_annotations["text"] = ' '.join(splits[4:])
                    annotations.append(file_annotations)

        annotations.sort(key=lambda x: x["start"])
        return annotations

    def get_text(self, txt_file: str) -> str:
        with open(txt_file, 'r') as f:
            text = f.read()

        return text
    
    def convert(self) -> None:
        nlp = English()
        tokenizer = nlp.tokenizer

        special_case = [{ORTH: "B"}, {ORTH: "."}]
        tokenizer.add_special_case("B.", special_case)

        prefixes = nlp.Defaults.prefixes #+ [r'-', r'±', r'≈']
        suffixes = nlp.Defaults.suffixes #+ [r')', r'(?<=B)\.']
        infixes = nlp.Defaults.infixes + [r'±', r'\+/-', r'-/\+', r'=', r'/(?=\d)', r':', r'--', r'\(', r'\)'] #'vs.' 
        prefix_regex = compile_prefix_regex(prefixes)
        suffix_regex = compile_suffix_regex(suffixes)
        infix_regex = compile_infix_regex(infixes)

        tokenizer.prefix_search = prefix_regex.search
        tokenizer.suffix_search = suffix_regex.search
        tokenizer.infix_finditer = infix_regex.finditer

        with open(self.output_file, 'w') as f:
            for ann_file, txt_file in self.get_file_pairs():

                BUGED = [
                    "brat_annotations/15023242.ann",
                    "brat_annotations/16520033.ann",
                    "brat_annotations/12796608.ann",
                    "brat_annotations/17785705.ann",
                ]
                if ann_file in BUGED:
                    continue

                annotations = self.get_annotations(ann_file)
                tokens = tokenizer(self.get_text(txt_file))
                len_annotations = len(annotations)

                annotation_idx = 0
                tag_start = annotations[annotation_idx]["start"]
                tag_end = annotations[annotation_idx]["end"]
                tag_span = tokens.char_span(tag_start, tag_end)
                for token in tokens:
                    if token not in tag_span:
                        f.write(f"{token} O\n")
                    else:
                        ner_tag = annotations[annotation_idx]["ner_tag"]
                        pos = "B" if token == tag_span[0] else "I"
                        f.write(f"{token} {pos}-{ner_tag}\n")
                        if token == tag_span[-1]:
                            annotation_idx += 1
                            if annotation_idx < len_annotations:
                                tag_start = annotations[annotation_idx]["start"]
                                tag_end = annotations[annotation_idx]["end"]
                                tag_span = tokens.char_span(tag_start, tag_end)
                                if tag_span is None:
                                    pprint(annotations); print()
                                    print(f"TEXT : {tokens.text}"); print()
                                    print(f"TOKENS : {[t.text for t in tokens]}"); print()
                                    print(f"In {ann_file} : token '{token}' at {tag_start}-{tag_end}, {tag_span}"); print()

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