from collections import defaultdict
import spacy
import tokenizations
import transformers

from tags import get_tag_replacers, match_replacers_with_nps, filter_words_from_span, _merge_that_with_nearest_np, _remove_how_about
from const import TAGS_REGEXP, DEPENDENT_PHRASE_REGEXP, F_DEPENDENT_PHRASE_REGEXP, NON_F_DEPENDENT_PHRASE_REGEXP, GROUP_NPS, EXISTENCE_NPS, THINGS, UNIQUE_NPS

nlp = spacy.load("en_core_web_sm")
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")


def resolve_caption(replaced_nps, objects_ids, template_info, i):
    groupped = defaultdict(list)
    focus_id = None
    non_focus_ids = None
    if len(replaced_nps) == len(objects_ids):
        for focus_id, (tag, replaced_np) in zip(objects_ids, replaced_nps.items()):
            groupped[focus_id].append({
                'np': replaced_np,
                'round': i
            })
        non_focus_ids = objects_ids[1:]
    if len(replaced_nps) == 1 and 'count' in template_info['label']:
        focus_id = '-'.join(map(str, objects_ids)) if len(objects_ids) > 1 else objects_ids[0]
        groupped[focus_id].append({
            'np': list(replaced_nps.values())[0],
            'round': i
        })
    return groupped, focus_id, non_focus_ids


def resolve_unique_caption(processed_text, objects_ids, i):
    groupped = defaultdict(list)
    focus_id = None
    non_focus_ids = None
    for np in UNIQUE_NPS:
        if np in processed_text.text.lower():
            start = processed_text.text.lower().find(np)
            focus_id = objects_ids[0] if objects_ids else None
            groupped[focus_id].append({
                'np': processed_text.char_span(start, start + len(np)), 'round': i
            })
            return groupped, focus_id, non_focus_ids
    return groupped, focus_id, non_focus_ids

def resolve_caption_things(template_info, dialog, objects_ids, things, focus_id, i):
    groupped = defaultdict(list)
    if 'extreme' in template_info['label']:
        if template_info['index'] == 2 or template_info['index'] == 3:
            dialog_objects = list(map(int, dialog['graph']['objects'].keys()))
            if focus_id in dialog_objects:
                dialog_objects.remove(focus_id)
            focus_id = '-'.join(map(str, dialog_objects)) if len(dialog_objects) > 1 else dialog_objects[0]
    if focus_id is None:
        focus_id = '-'.join(map(str, objects_ids)) if len(objects_ids) > 1 else objects_ids[0]
    groupped[focus_id].append({
        'np': things[0],
        'round': i
    })
    return groupped, focus_id


def resolve_non_focus(dialog, replaced_nps, objects_ids, template_info, i, dependence, round_focus_ids,
                      round_non_focus_ids):
    groupped = defaultdict(list)
    non_focus_id = None
    focus_id = None

    non_dependent_nps = {t: r for t, r in replaced_nps.items() if TAGS_REGEXP.match(t)}
    dependent_nps = {t: r for t, r in replaced_nps.items() if DEPENDENT_PHRASE_REGEXP.match(t)}
    if len(objects_ids) == len(non_dependent_nps):
        if len(objects_ids) == 1 and 'count' in template_info['label']:
            # use answer as the reference to count questions
            groupped[objects_ids[0]].append({
                'answer': True,
                'np': nlp(str(dialog['dialog'][i - 1]['answer'])),
                'round': i}
            )
            focus_id = objects_ids[0]
        else:
            # match each non-dependent np with the object
            for o_id, (tag, replaced_np) in zip(objects_ids, non_dependent_nps.items()):
                groupped[o_id].append({
                    'np': replaced_np,
                    'round': i
                })
                non_focus_id = o_id
    if dependence is not None and len(dependent_nps) == 1:
        # check if the previous dependence is the last one, take the the focus id from it
        if dialog['graph']['history'][dependence + 1].get('dependence') is None:
            focus_id = round_focus_ids[dependence + 1]
            groupped[focus_id].append({
                'np': list(dependent_nps.values())[0],
                'round': i
            })
        # for counting questions check the non-focus objects
        elif 'count' in dialog['template_info'][dependence + 1]['label']:
            if round_non_focus_ids[dependence + 1] is not None:
                group_id = round_non_focus_ids[dependence + 1]
                groupped[group_id].append({
                    'np': list(dependent_nps.values())[0],
                    'round': i
                })
    return groupped, focus_id, non_focus_id


def _prepare_nps(dialog, i):
    text = dialog['dialog'][i - 1]['question'] if i > 0 else dialog['caption']
    tag_replacers = get_tag_replacers(text, dialog['template_info'][i])
    processed_text = nlp(text)
    nps = list(processed_text.noun_chunks)
    nps = _merge_that_with_nearest_np(nps)
    nps = _remove_how_about(nps)
    replaced_nps = match_replacers_with_nps(tag_replacers, nps, processed_text)
    return replaced_nps, nps, processed_text


def resolve_counting(objects_ids, dialog, i):
    group_id = '-'.join(map(str, objects_ids)) if len(objects_ids) > 1 else objects_ids[0]
    reference = {
        'answer': True,
        'np': nlp(str(dialog['dialog'][i - 1]['answer'])),
        'round': i}
    return group_id, reference


def group_nps_by_referents(dialog):
    groupped = defaultdict(list)
    round_focus_ids = []
    round_non_focus_ids = []
    for history in dialog['graph']['history']:
        i = history['round']
        template_info = dialog['template_info'][i]
        dependence = history.get('dependence', None)
        focus_id = history.get('focus_id')
        objects_ids = [o['id'] for o in history['objects']]
        replaced_nps, question_nps, processed_text = _prepare_nps(dialog, i)
        things = [np for np in question_nps if any(w in np.text for w in THINGS) and np not in replaced_nps.values()]

        non_focus_id = None
        # caption
        if i == 0:
            if 'unique' in template_info['label'] and template_info.get('index', -1) == 2:
                caption_groupped, focus_id, non_focus_id = resolve_unique_caption(processed_text, objects_ids, i)
            else:
                caption_groupped, focus_id, non_focus_id = resolve_caption(replaced_nps, objects_ids, template_info, i)
            for g, nps in caption_groupped.items():
                groupped[g].extend(nps)

            if len(things) == 1:
                things_groupped, non_focus_id = resolve_caption_things(template_info, dialog, objects_ids, things, focus_id, i)
                for g, nps in things_groupped.items():
                    groupped[g].extend(nps)
            round_focus_ids.append(focus_id)
            round_non_focus_ids.append(non_focus_id)
            continue

        if focus_id is None:
            non_focus_group, focus_id, non_focus_id = resolve_non_focus(dialog, replaced_nps, objects_ids,
                                                                         template_info, i, dependence,
                                                                         round_focus_ids,
                                                                         round_non_focus_ids)

            if focus_id is not None and len(replaced_nps) == 1:
                groupped[focus_id].append({'np': list(replaced_nps.values())[0], 'round': i})
            round_focus_ids.append(focus_id)
            for g, nps in non_focus_group.items():
                groupped[g].extend(nps)

            if focus_id in objects_ids:
                objects_ids.remove(focus_id)
            if 'count' in template_info['label'] and len(objects_ids) > 0:
                group_id, reference = resolve_counting(objects_ids, dialog, i)
                groupped[group_id].append(reference)
                non_focus_id = group_id
                if len(replaced_nps) == 1:
                    groupped[group_id].append({'np': list(replaced_nps.values())[0], 'round': i})
                elif len(things) == 1:
                    groupped[group_id].append({'np': things[0], 'round': i})
            elif 'attribute-group' in template_info['label'] and dependence is not None:
                group_np = [np for np in question_nps if any(np.text.lower() == t for t in GROUP_NPS)]
                group_id = round_non_focus_ids[dependence + 1]
                group_id = group_id if group_id is not None else round_focus_ids[dependence + 1]
                if len(group_np) == 1:
                    already_groupped = any(
                        group_np[0] == np['np']
                        for np in groupped[group_id]
                        if isinstance(np['np'], spacy.tokens.Span)
                    )
                    if not already_groupped:
                        groupped[group_id].append({
                            'np': group_np[0],
                            'round': i
                        })
                        non_focus_id = group_id
            elif len(objects_ids) > 0 and len(replaced_nps) == 1:
                non_focus_id = '-'.join(map(str, objects_ids)) if len(objects_ids) > 1 else objects_ids[0]
                groupped[non_focus_id].append({
                    'np': list(replaced_nps.values())[0],
                    'round': i
                })
            round_non_focus_ids.append(non_focus_id)
            continue

        for tag, np in replaced_nps.items():
            if F_DEPENDENT_PHRASE_REGEXP.match(tag):
                groupped[focus_id].append({
                    'np': np,
                    'round': i
                })
            elif NON_F_DEPENDENT_PHRASE_REGEXP.match(tag) and len(objects_ids) >= 1:
                groupped[objects_ids[0]].append({
                    'np': np,
                    'round': i
                })
                if len(things) == 1:
                    groupped[objects_ids[0]].append({
                        'np': things[0],
                        'round': i
                    })
        # for counting questions
        if focus_id in objects_ids:
            objects_ids.remove(focus_id)
        if 'count' in template_info['label'] and len(objects_ids) > 0:
            group_id, reference = resolve_counting(objects_ids, dialog, i)
            groupped[group_id].append(reference)
            non_focus_id = group_id
            if len(things) == 1:
                groupped[group_id].append({
                    'np': things[0],
                    'round': i
                })
        # for the existence questions
        if 'exist-obj' in template_info['label'] and len(objects_ids) > 0:
            non_focus_id = '-'.join(map(str, objects_ids)) if len(objects_ids) > 1 else objects_ids[0]
            group_np = [np for np in question_nps if any(np.text.lower() == t for t in EXISTENCE_NPS)]
            if len(group_np) == 1:
                group_np = group_np[0]
                groupped[non_focus_id].append({
                    'np': group_np,
                    'round': i,
                    'existence_followed': False
                })
            else:
                tokens = [t for t in EXISTENCE_NPS if t in processed_text.text.lower()]
                for t in tokens:
                    start = processed_text.text.lower().index(t)
                    end = start + len(t)
                    group_np = processed_text.char_span(start,end)
                    groupped[non_focus_id].append({
                        'np': group_np,
                        'round': i,
                        'existence_followed': False
                    })
        round_focus_ids.append(focus_id)
        round_non_focus_ids.append(non_focus_id)
    return groupped


def filter_solo_non_pronoun_nps(groupped_nps):
    filtered_groups = {}
    for group_id, nps in groupped_nps.items():
        if len(nps) == 1 and not any(t.pos_ == 'PRON' for np in nps for t in np['np']):
            continue
        if len(nps) == 1 and all(t.pos_ == 'PRON' for np in nps for t in np['np']):
            continue
        filtered_groups[group_id] = nps
    return filtered_groups


def filter_how_many_nps(groupped_nps):
    filtered_groups = {}
    for group_id, nps in groupped_nps.items():
        filtered_nps = []
        for np in nps:
            if 'existence_followed' in np:
                if not np['existence_followed']:
                    continue
            filtered_nps.append(np)
        if filtered_nps:
            filtered_groups[group_id] = filtered_nps
    return filtered_groups


def filter_right_nps(groupped_nps):
    filtered_groups = {}
    for group_id, nps in groupped_nps.items():
        filtered_nps = []
        for np in nps:
            np_span = filter_words_from_span('right', ['right'], np['np'])
            if len(np_span) == 0:
                continue
            np['np'] = np_span
            filtered_nps.append(np)
        if filtered_nps:
            filtered_groups[group_id] = filtered_nps
    return filtered_groups


def filter_existence(groupped_nps):
    filtered_groups = {}
    for group_id, nps in groupped_nps.items():
        filtered_nps = []
        for np in nps:
            np_span = filter_words_from_span('how many', ['how', 'many'], np['np'])
            if len(np_span) == 0:
                continue
            np['np'] = np_span
            filtered_nps.append(np)
        if filtered_nps:
            filtered_groups[group_id] = filtered_nps
    return filtered_groups


def count_start_per_round(dialog, tokens=True):
    ix = 1 if tokens else 0
    start_per_round = [ix]
    caption = dialog['caption']
    if tokens:
        caption = tokenizer.tokenize(caption)
    ix += len(caption) + 1
    for r in dialog['dialog']:
        start_per_round.append(ix)
        q = r['question']
        a = str(r['answer'])
        if tokens:
            q = tokenizer.tokenize(q)
            a = tokenizer.tokenize(a)
        ix += len(q) + 1 + len(a) + 1
    return start_per_round, ix - 1


def _get_difference(mapping):
    pad = 0
    difference = []
    for i, m in enumerate(mapping):
        if len(m) == 1 and m[0] == i + pad:
            difference.append(pad)
            continue
        for i_m in m:
            if i_m == i + pad:
                difference.append(pad)
                continue
            pad += 1
    difference.append(pad)
    return difference


def get_aligned_rounds(dialog):
    rounds = []
    caption_bert = tokenizer.tokenize(dialog['caption'])
    caption_spacy = [t.lower_ for t in nlp(dialog['caption'])]
    spacy2bert, _ = tokenizations.get_alignments(caption_spacy, caption_bert)
    difference = _get_difference(spacy2bert)
    rounds.append(difference)
    for r in dialog['dialog']:
        q_bert = tokenizer.tokenize(r['question'])
        q_spacy = [t.lower_ for t in nlp(r['question'])]
        spacy2bert, _ = tokenizations.get_alignments(q_spacy, q_bert)
        difference = _get_difference(spacy2bert)
        rounds.append(difference)
    return rounds


def index_clusters(groupped_nps, start_per_round, end_ix, aligned_rounds=[], last_round=-1, tokens=True):
    clusters = []
    if last_round == -1:
        last_round = len(start_per_round) - 1
    for group, nps in groupped_nps.items():
        cluster = []
        for np in nps:
            r = np['round']
            if r > last_round:
                continue
            span = np['np']
            if np.get('answer', False):
                if r == last_round:
                    continue
                end = start_per_round[r + 1] - 2 if r < len(start_per_round) - 1 else end_ix - 1
                start = end - len(span) + 1
                cluster.append([start, end])
                continue
            start = span.start_char
            end = span.end_char
            if tokens:
                difference = aligned_rounds[r]
                start = span.start
                start += difference[start]
                end = span.end - 1
                end += difference[end + 1]

            start += start_per_round[r]
            end += start_per_round[r]
            cluster.append([start, end])
        if cluster:
            clusters.append(cluster)
    clusters = [cluster for cluster in clusters if len(cluster) > 1]
    return clusters


def group_nps_pipline(dialog):
    groupped_nps = group_nps_by_referents(dialog)
    groupped_nps = filter_right_nps(groupped_nps)
    groupped_nps = filter_solo_non_pronoun_nps(groupped_nps)
    groupped_nps = filter_existence(groupped_nps)
    return groupped_nps


def extract_clusters(dialog):
    groupped_nps = group_nps_pipline(dialog)
    start_token_per_round, end_ix, aligned_rounds = prepare_for_indexing(dialog)
    clusters = index_clusters(groupped_nps, start_token_per_round, end_ix, aligned_rounds)
    return clusters


def prepare_for_indexing(dialog):
    start_token_per_round, end_ix = count_start_per_round(dialog)
    aligned_rounds = get_aligned_rounds(dialog)
    return start_token_per_round, end_ix, aligned_rounds
