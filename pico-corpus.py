import csv
import json
import os

import datasets


_CITATION = """\
@InProceedings{mutinda2022pico,
  title = {PICO Corpus: A Publicly Available Corpus to Support Automatic Data Extraction from Biomedical Literature},
  author = {Mutinda, Faith and Liew, Kongmeng and Yada, Shuntaro and Wakamiya, Shoko and Aramaki, Eiji},
  booktitle = {Proceedings of the first Workshop on Information Extraction from Scientific Publications},
  pages = {26--31},
  year = {2022}
}
"""

_DESCRIPTION = """\
The corpus consists of about 1,011 PubMed abstracts which are RCTs related
to breast cancer. For each abstract, text snippets that identify the
Participants, Intervention, Control, and Outcome (PICO elements) are annotated.
The abstracts were annotated using BRAT (https://brat.nlplab.org/) and later
converted to CoNLL-2003.
"""

_HOMEPAGE = "https://github.com/sociocom/PICO-Corpus"

_URL = "https://github.com/Martin-Masson/pico-corpus"

_TAGS = [
    "O",
    "total-participants",
    "intervention-participants",
    "control-participants",
    "age",
    "eligibility",
    "ethinicity",
    "condition",
    "location",
    "intervention",
    "control",
    "outcome",
    "outcome-Measure",
    "iv-bin-abs",
    "cv-bin-abs",
    "iv-bin-percent",
    "cv-bin-percent",
    "iv-cont-mean",
    "cv-cont-mean",
    "iv-cont-median",
    "cv-cont-median",
    "iv-cont-sd",
    "cv-cont-sd",
    "iv-cont-q1",
    "cv-cont-q1",
    "iv-cont-q3",
    "cv-cont-q3",
]


class PicoCorpus(datasets.GeneratorBasedBuilder):
    """A corpus of about 1,011 PubMed abstracts from RCTs related to breast cancer"""

    VERSION = datasets.Version("1.1.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"), 
                    "tokens": datasets.Sequence(datasets.Value("string")), 
                    "ner_tags": datasets.Sequence(datasets.ClassLabel(names=_TAGS)),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.ALL,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "pico_conll.txt"),
                },
            )
        ]

    def _generate_examples(self, filepath):
        idx = 0
        tokens = []
        ner_tags = []
        labels = {tag: i for i, tag in enumerate(_TAGS)}

        with open(filepath) as f:
            lines = f.read().splitlines()
            for line in lines:
                if not line:
                    yield id, {
                        "id": str(idx),
                        "tokens": tokens,
                        "ner_tags": ner_tags
                    }
                    idx += 1
                    tokens.clear()
                    ner_tags.clear()
                else:
                    token_tag = line.split()
                    tokens.append(token_tag[0])
                    ner_tags.append(labels[token_tag[1]])
