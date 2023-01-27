import re
import os
import json
import spacy
import difflib
import transformers

from const import (
    TAGS_SPLIT_REGEXP, FIGURE_BRACKETS_REGEXP, WHITESPACE_PUNCT_REGEXP, THING_TO_OBJECT, OPTIONAL_TEXT_REGEXP,
    DEPENDENT_PHRASE_REGEXP, TAGS_REGEXP
)

nlp = spacy.load("en_core_web_sm")
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

meta_info = json.load(open(f'ContextVD/data_construction/templates/metainfo.json'))
synonyms = json.load(open(f'ContextVD/data_construction/templates/synonyms.json'))

question_templates = []
for f in os.listdir('ContextVD/data_construction/templates/questions'):
    t = json.load(open(f'ContextVD/data_construction/templates/questions/{f}'))
    question_templates.extend(t)
question_templates = {t['label']: t for t in question_templates}


def split_template(text):
    groups = TAGS_SPLIT_REGEXP.split(text)
    new_tokens = []
    for group in groups:
        if FIGURE_BRACKETS_REGEXP.match(group):
            new_tokens.append(group)
            continue
        new_tokens.extend(WHITESPACE_PUNCT_REGEXP.split(group))
    return new_tokens


def compare_tags_to_text(text_template, text_sample):
    d = difflib.Differ()
    diff = d.compare(split_template(text_template), WHITESPACE_PUNCT_REGEXP.split(text_sample))
    diff = list(diff)

    tag_replacers = {}
    tag = None
    replacers = []
    char_ix = 0
    for token in diff:
        if token.startswith('- '):
            tag = token[2:] if tag is None else f'{tag}-{token[2:]}'
        elif token.startswith('+ '):
            replacer = token[2:]
            if replacer:
                replacers.append([replacer, char_ix, char_ix + len(replacer)])
            char_ix += len(token) - 1
        elif token.startswith('? '):
            continue
        elif tag is not None:
            char_ix += len(token) - 1
            if not replacers:
                tag = None
                continue
            tag_replacers[tag] = replacers
            tag = None
            replacers = []
        else:
            char_ix += len(token) - 1

    if tag is not None and replacers:
        tag_replacers[tag] = replacers
    return tag_replacers


def split_synonyms(words_set, words_set_no_spans, synonyms):
    synonyms = sorted([s.split() for s in synonyms], key=len, reverse=True)
    for synonym in synonyms:
        ixs = [words_set_no_spans.index(s) for s in synonym if s in words_set_no_spans]
        if len(ixs) == len(synonym):
            synonym = [words_set[i] for i in ixs]
            return synonym


def compare_with_synonyms(words_target, words_set, synonyms=synonyms, synonyms_first=False):
    diff = []
    words_set_no_spans = [w[0] for w in words_set]
    for w_t in words_target:
        match = None
        if synonyms_first and w_t in synonyms:
            match = split_synonyms(words_set, words_set_no_spans, synonyms[w_t] + [w_t])
        elif w_t in words_set_no_spans:
            match = [words_set[words_set_no_spans.index(w_t)]]
        elif w_t in synonyms:
            match = split_synonyms(words_set, words_set_no_spans, synonyms[w_t])
        if match:
            diff.extend(match)
    return diff


def separate_merged_tags(tag_replacers, meta_info=meta_info):
    new_tag_replacers = {}
    for tags, replacers in tag_replacers.items():
        tags = tags.split('-')
        things = [i for i, pair in enumerate(THING_TO_OBJECT) if all(t in tags for t in pair[0])]
        for i in things:
            pair = THING_TO_OBJECT[i]
            tags = [t for t in tags if t not in pair[0]]
            replacers = [t for t in replacers if t[0] not in pair[1]]
        tags_leftover = []
        for tag in tags:
            if OPTIONAL_TEXT_REGEXP.match(tag):
                words_in_brackets = tag[1:-1].split()
                overlap = compare_with_synonyms(words_in_brackets, replacers)
            elif tag == '<R>':
                overlap = compare_with_synonyms(
                    meta_info['relations'],
                    replacers,
                    synonyms=meta_info['relation_phrases'],
                    synonyms_first=True
                )
            elif tag == '<P>':
                overlap = compare_with_synonyms(meta_info['relations'], replacers)
            elif tag == '<A>':
                overlap = compare_with_synonyms(meta_info['attributes'], replacers)
            else:
                tags_leftover.append(tag)
                continue
            replacers = [w for w in replacers if w not in overlap]
            new_tag_replacers[tag] = overlap
        if tags_leftover:
            new_tag_replacers[' '.join(tags_leftover)] = replacers
    return new_tag_replacers


def get_tag_replacers(text_sample, template, print_text=False):
    if 'references' in template:
        text_template = template['references'][template['index']]
    elif template['inputs'] == 0:
        text_template = text_sample
    else:
        q_t = question_templates[template['label']]
        if 'references' in q_t:
            text_template = q_t['references'][template['index']]
        else:
            text_template = q_t['text'][template['index']]
    if print_text:
        print(text_sample + '\n' + text_template)
    tag_replacers = compare_tags_to_text(text_template, text_sample)
    tag_replacers = separate_merged_tags(tag_replacers)
    return tag_replacers


def _merge_that_with_nearest_np(nps):
    new_nps = []
    for i, np in enumerate(nps):
        if np.text.lower() == 'that':
            if i + 1 < len(nps) and np.end == nps[i + 1].start:
                merge = np.doc[np.start:nps[i + 1].end]
                nps[i + 1] = merge
                continue
        new_nps.append(np)
    return new_nps


def _remove_how_about(nps):
    new_nps = []
    for np in nps:
        np = filter_words_from_span('how about', ['how','about'], np)
        new_nps.append(np)
    return new_nps


def filter_words_from_span(phrase, words, span):
    if phrase in span.text.lower():
        tokens = [t.i for t in span if t.lower_ not in words]
        if not tokens:
            return span
        start = min(tokens)
        end = max(tokens)
        span = span.doc[start:end + 1]
    return span

def interval_overlap(interval0, interval1):
    start = max([interval0[0], interval1[0]])
    end = min([interval0[1], interval1[1]])
    return end - start > 0

def is_inside(interval0, interval1):
    return interval1[1] >= interval0[0] >= interval1[0] and interval1[0] <= interval0[1] <= interval1[1]

def match_replacers_with_nps(tag_replacers, nps, doc):
    replaced_nps = {}
    replacers_ranges = {
        tag: [min(r[1] for r in replacers), max(r[2] for r in replacers)]
        for tag, replacers in tag_replacers.items()
        if replacers
    }
    for tag, replacers in tag_replacers.items():
        if DEPENDENT_PHRASE_REGEXP.match(tag) or TAGS_REGEXP.match(tag):
            replaced_start = replacers_ranges[tag][0]
            replaced_end = replacers_ranges[tag][1]
            replaced_np = [np for np in nps if replaced_start >= np.start_char and replaced_end <= np.end_char]
            if replaced_np:
                replaced_np = replaced_np[0]
                if 'PRP$' in [t.tag_ for t in replaced_np]:
                    i = [t.tag_ for t in replaced_np].index('PRP$')
                    replaced_np = replaced_np[i:i + 1]
                # replaced_span = [replaced_np.start_char, replaced_np.end_char]
                # overlap = [(r_range, tag) for tag, r_range in replacers_ranges.items() if interval_overlap(r_range, replaced_span)]
                # overlap = [o for o in overlap if not is_inside(o[0], replaced_span)]
                # if overlap:
                #     replaced_np = replaced_np
                #     # replaced_np = replaced_np.char_span(*overlap[0])
                replaced_nps[tag] = replaced_np
            else:
                replaced_start = min(r[1] for r in replacers)
                replaced_end = max(r[2] for r in replacers)
                np = doc.char_span(replaced_start, replaced_end)
                replaced_nps[tag] = np
    return replaced_nps
