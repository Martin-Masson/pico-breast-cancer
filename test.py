from spacy.lang.en import English
import spacy


file = "brat_annotations/31568587.txt"
text = ""
with open(file, 'r') as f:
    text = f.read()

print(text[526:595])

"""
nlp = English()
tokenizer = nlp.tokenizer
infixes = nlp.Defaults.infixes + [r'='] + [r'--']
infixes_regex = spacy.util.compile_infix_regex(infixes)
tokenizer.infix_finditer = infixes_regex.finditer

tokens = tokenizer(text)
print(tokens.text)
print()
print([t.text for t in tokens])
print()

tag_start = 562
tag_end = 565
tag_span = tokens.char_span(tag_start, tag_end)
tag_span_text = tokens.text[tag_start:tag_end]

print(f"{tag_start}, {tag_end}: {tag_span} | {tag_span_text}")
"""