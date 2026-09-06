from os import path as osp

from basicsr.utils import scandir


def paired_paths_from_meta_info_file(folders, keys, meta_info_file, filename_tmpl):
    """Generate paired LQ/GT entries from a meta-info file."""
    assert len(folders) == 2, (
        f'The len of folders should be 2 with [input_folder, gt_folder], got {len(folders)}.')
    assert len(keys) == 2, (
        f'The len of keys should be 2 with [input_key, gt_key], got {len(keys)}.')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    with open(meta_info_file, 'r') as fin:
        gt_names = [line.split(' ')[0] for line in fin]

    paths = []
    for gt_name in gt_names:
        basename, ext = osp.splitext(osp.basename(gt_name))
        input_name = f'{filename_tmpl.format(basename)}{ext}'
        paths.append({
            f'{input_key}_path': osp.join(input_folder, input_name),
            f'{gt_key}_path': osp.join(gt_folder, gt_name),
        })
    return paths


def paired_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired LQ/GT entries by scanning two image folders."""
    assert len(folders) == 2, (
        f'The len of folders should be 2 with [input_folder, gt_folder], got {len(folders)}.')
    assert len(keys) == 2, (
        f'The len of keys should be 2 with [input_key, gt_key], got {len(keys)}.')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_paths = sorted(list(scandir(input_folder)))
    gt_paths = sorted(list(scandir(gt_folder)))
    assert len(input_paths) == len(gt_paths), (
        f'{input_key} and {gt_key} datasets have different number of images: '
        f'{len(input_paths)}, {len(gt_paths)}.')

    paths = []
    for idx, gt_path in enumerate(gt_paths):
        basename, _ = osp.splitext(osp.basename(gt_path))
        _, input_ext = osp.splitext(osp.basename(input_paths[idx]))
        input_name = f'{filename_tmpl.format(basename)}{input_ext}'
        assert input_name in input_paths, f'{input_name} is not in {input_key}_paths.'
        paths.append({
            f'{input_key}_path': osp.join(input_folder, input_name),
            f'{gt_key}_path': osp.join(gt_folder, gt_path),
        })
    return paths
