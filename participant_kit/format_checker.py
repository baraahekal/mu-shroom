import pathlib
import argparse

from scorer import load_jsonl_file_to_records


def try_load(filename, is_ref=False):
    assert filename.is_file(), f'No such file: {filename}' 
    try:
        data = load_jsonl_file_to_records(filename, is_ref=is_ref)
    except Exception as exc:
        raise RuntimeError(f"Couldn't read file {filename}") from exc
    return data


def get_reference_data(ref_dir, lang):
    return try_load(ref_dir / f'mushroom.{lang}-tst.v1.jsonl', is_ref=True)


def check_aligned(pdict, rdict, fname):
    if pdict['id'] != rdict['id']:
        raise RuntimeError(f'IDs are not correctly aligned in {fname}: {pdict["id"]} != {rdict["id"]}')
    if 'hard_labels' in pdict:
        for start, end in pdict['hard_labels']:
            if not (0 <= start < end <= rdict['text_len']):
                 raise RuntimeError(f'hard prediction {pdict["id"]} concerns chars not in range (or ends before it starts).')
    if 'soft_labels' in pdict:
        for lbl in pdict['soft_labels']:
            start, end = lbl['start'], lbl['end']
            if not (0 <= start < end <= rdict['text_len']):
                 raise RuntimeError(f'soft prediction {pdict["id"]} concerns chars not in range  (or ends before it starts).')


def main(preds_file, ref_dir):
    seen_langs = set()
    for fname in preds_file:
        pred_dicts = try_load(fname)
        if not fname.name.endswith('.jsonl'):
            raise RuntimeError(f'All files should be in the .jsonl format, but you submitted {fname.name}.')
        pred_dicts = try_load(fname)
        split, lang, _ = pred_dicts[0]['id'].split('-')
        if split != 'tst':
            raise RuntimeError(f"The format checker is only configured for test preds.")
        ref_dicts = get_reference_data(ref_dir, lang)
        for pdict, rdict in zip(pred_dicts, ref_dicts):
            check_aligned(pdict, rdict, fname.name)
        if lang in seen_langs:
            raise RuntimeError(f'Multiple files for language {lang}')
        seen_langs.add(lang)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path_to_unlabeled_test_files', 
        type=pathlib.Path,
        help='Path pointing to a directory containing the JSONL files for the'\
        'unlabeled test split. Necessary so that '
    )
    parser.add_argument('jsonl_files', nargs='+', type=pathlib.Path)
    args = parser.parse_args()
    main(args.jsonl_files, args.path_to_unlabeled_test_files)
    print('all good')
