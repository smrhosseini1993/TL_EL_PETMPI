# Data format

The study data are not included because they contain patient information. The scripts expect the following local structure:

```text
data/
├── training/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ica_lables.txt
├── test/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ica_lables.txt
├── clinical_baselines.csv                 # local only; optional baseline input
├── clinical_reader_calls.csv              # local only; optional reader comparator input
├── results_TL.xlsx                        # local only
├── results_CNN.xlsx                       # local only
├── results_EL.xlsx                        # local only
└── results_baselines.xlsx                 # local only
```

## Polar-map images and labels

Images must be RGB JPEG polar maps that have already been cropped to the polar-map field of view. The training scripts decode each JPEG, resize it to the selected input size (128 × 128 in the study launcher), and scale RGB values to the 0–1 range by division by 255. They do not perform the cropping step.

Each label file contains one binary CAD label (`0` or `1`) per line. Labels are matched to the alphabetical order of the corresponding image filenames. For the fixed-split workflow, the alphabetically sorted `training` images are divided into the first 61 images for training and the remaining 31 for validation; the `test` folder contains 46 images. The cross-validation workflow reads only the 92 `training` images. File names—not patient names—define the saved order used by the local result workbooks and notebooks.

## Local comparator and reader inputs

`clinical_baselines.csv` is a local 138-row file for `baselines.py`. It contains a unique study identifier, a binary CAD label, 17 segmental stress-MBF columns (`STR 1` through `STR 17`), and the clinical variables listed by `python baselines.py --help`. Its row order must be 61 training, 31 validation, and 46 test patients.

`clinical_reader_calls.csv` is an optional local two-column file for the reader comparison in `notebooks/evaluate_results.ipynb`:

```text
within_test_order,clinical_reader_call
1,0
2,1
```

It contains one binary clinical-reader call per test patient in the saved filename order. It does not contain reader probability scores or patient names.

## Privacy and version control

Do not add images, label files, clinical data, participant identifiers, result workbooks, prediction vectors, model files, or generated figures to a public repository. The project `.gitignore` excludes these local inputs and outputs by default.
