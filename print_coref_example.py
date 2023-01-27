import json 
from ContextVD.data_construction.clevr_utils import pretty_print_corefs

dialog = json.load(open('coref_dialog_to_print_example.json'))
groups = json.load(open('coref_groups_to_print_example.json'))
groups = {int(i): group for i, group in groups.items()}
pretty_print_corefs(dialog, groups)
