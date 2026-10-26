# Training split indices and historical generation utility

The `.npy` files in `kitti/` and `nusc/` contain the released training splits.
Use these files directly when reproducing the experiments. Each file contains
scan-level indices for the labeled subset; the training loader constructs the
unlabeled subset from the remaining training indices.

For SemanticKITTI, indices refer to the scan list obtained by iterating over
`split_sequence.train` in the dataset YAML and concatenating the sorted filenames
within each sequence. For nuScenes, indices refer to the ordering of `data['infos']`
in the training info pickle used by the loader. Preserve these orderings when
using the released indices.

| Dataset | Novel class IDs | Labeled scans | Index file |
| --- | --- | ---: | --- |
| SemanticKITTI | 4, 6 | 9,565 | `kitti/selected_idx_in__ex_4_6_50%.npy` |
| SemanticKITTI | 2, 4, 5 | 8,608 | `kitti/selected_idx_in__ex_2_4_5_45%.npy` |
| SemanticKITTI | 3, 6, 9 | 7,652 | `kitti/selected_idx_in__ex_3_6_9_40%.npy` |
| nuScenes | 1, 8 | 12,658 | `nusc/nusc_selected_idx_in__ex_1_8_45%.npy` |
| nuScenes | 1, 2, 5 | 9,845 | `nusc/nusc_selected_idx_in__ex_1_2_5_35%.npy` |
| nuScenes | 0, 4, 7, 8 | 7,032 | `nusc/nusc_selected_idx_in__ex_0_4_7_8_25%.npy` |

Class IDs use the dataset YAML's `learning_map` label space.

## Historical candidate-selection code

[`class_stats_v2.py`](../class_stats_v2.py) is an unchanged historical source file
recovered from the original development repository (commit `6ef69de`). It builds
a per-scan class-presence matrix and selects scans satisfying include/exclude
class constraints. To construct labeled candidates, leave the include classes
empty and exclude the split's novel classes. Scans containing any excluded class
are omitted from the candidate set.

This utility does **not** perform the final labeled-ratio adjustment. The separate
historical `class_num_rebalance.py` source has not been recovered. Running this
utility alone will therefore not regenerate the released split files exactly.

## Using the historical utility

This is an archived interactive script rather than a command-line split generator.
Run it from the repository root in the project's environment, with `numba` and
`tqdm` available. Before running it, adjust its hardcoded settings:

1. In the `__main__` block, set `dataset` to `'kitti'` or `'nusc'` and
   `data_split` to `'train'`. The archived default is the validation split.
2. Change the corresponding `dataset_config_path` to
   `config/semkitti_minkunet.yaml` or `config/nuscenes_minkunet.yaml`, and set
   `dataset_path` in that YAML to your dataset location.
3. For nuScenes, check the `origin_nuscenes_infos_*.pkl` paths in
   `nuScenesRestrictedDataset`. The info records must match the training loader's
   records and ordering; update the paths to your prepared info files as needed.
4. Run `python class_stats_v2.py`. If the class-presence cache does not exist,
   the script creates `class_stats.npy` or `class_stats_nusc.npy` and exits.
5. Run it again to select candidates interactively. For SemanticKITTI split 1,
   press Enter for include classes, enter `4 6` for exclude classes, and enter `y`
   to save `selected_idx_in__ex_4_6.npy` in the current directory. Stop the
   interactive loop with Ctrl+C when finished.

The resulting candidate index file has no ratio suffix and must not replace the
released final index files when reproducing the experiments. The existing dataset
loaders already implement the labeled/unlabeled partition from a final index file.
