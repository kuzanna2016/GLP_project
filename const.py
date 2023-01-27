import re

TAGS_SPLIT_REGEXP = re.compile(r'(?:\s|^)(?=\[|\{\w?\{)|(?<=\}\})(?:\s|$|[.,?])|(?<=\])(?:\s|$|[.,?])')
FIGURE_BRACKETS_REGEXP = re.compile(r'^\{\w?\{|^\[|\]$')
WHITESPACE_PUNCT_REGEXP = re.compile(r'[\s.,?]')
OPTIONAL_TEXT_REGEXP = re.compile(r'^\[.*?\]$')
DEPENDENT_PHRASE_REGEXP = re.compile(r'^\{\w?\{.*?\}\}$')
F_DEPENDENT_PHRASE_REGEXP = re.compile(r'^\{F\{.*?\}\}$')
NON_F_DEPENDENT_PHRASE_REGEXP = re.compile(r'^\{\{.*?\}\}$')
TAGS_REGEXP = re.compile(r'^(?:<\w>\s?){4}$')

THING_TO_OBJECT = [
    [['a', 'thing'], ['an', 'object']],
    [['things'], ['objects']]
]

THINGS = [
    'thing',
    'object',
]

GROUP_NPS = (
    'them',
    'the group',
    'this group',
)

EXISTENCE_NPS = (
    'other things',
    'other objects',
    'things',
    'objects',
)

UNIQUE_NPS = (
    'exactly one',
    'one'
)