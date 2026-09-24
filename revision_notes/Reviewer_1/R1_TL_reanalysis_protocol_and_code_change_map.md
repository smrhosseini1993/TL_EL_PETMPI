# R1 Transfer-Learning Reanalysis Protocol and Code-Change Map

**Manuscript:** *Transfer Learning and Ensemble Methods for Myocardial Ischemia Classification from 15O-H2O PET Polar Maps* (EJPH-D-26-00179)
**Scope:** Reviewer 1 comments affecting the transfer-learning (TL) workflow, principally Comments 3, 4, and 8.
**Status:** **Working protocol agreed in discussion; no revised TL code or GPU experiment has been run from this document.** This document is the single working source for the revised TL Methods text, response-to-reviewer text, and implementation checklist. It deliberately leaves ensemble construction, conventional PET baselines, clinical labels, and final patient-level statistics to their corresponding workstreams.

> **Core design decision.** The revision preserves the submitted legacy classifier head and the historic 61/31/46 comparison design, but repeats the TL benchmark with correct architecture-specific preprocessing, reproducible patient-level outputs, a development-only common hyperparameter search, and technically valid progressive fine-tuning.

---

## 1. Why the TL benchmark must be repeated

Reviewer 1 identified one direct methodological defect: all ImageNet-pretrained models were supplied with the same generic `image / 255.0` transformation. Keras Applications provide architecture-specific `preprocess_input` procedures because the pretrained ImageNet weights expect different channel conventions and value transformations. The reviewer explicitly states that the benchmarking experiment should be repeated if those procedures were not used.

The reanalysis therefore corrects **preprocessing**, not the research question. It retains the intended benchmark: eleven ImageNet-pretrained CNN backbones, one common retained classifier-head design, one historic final 61/31/46 split for direct comparison with the Teuho reference CNN, and 100 final random-seed runs for descriptive stability. The published Teuho reference CNN used cropped **256×256 RGB** polar maps, so revised TL inputs will also be 256×256 RGB.[1]

The revised study must not claim that the 100 runs represent 100 independent clinical samples. They remain repeated stochastic trainings on the same patients. Their role is descriptive stability reporting; patient-level inference will later use one locked aggregate prediction per test patient.[2]

---

## 2. Locked TL design

### 2.1 Benchmark question

The revised primary benchmark question is:

> **Under one development-selected common training protocol, how do the eleven prespecified ImageNet-pretrained CNN architectures compare on the independent 46-patient test cohort?**

This is a standardised-recipe comparison. It does **not** claim that every architecture has received unlimited bespoke optimisation. Every backbone receives the same candidate protocols, the same development folds, the same training budget, the same selection criterion, and no extra exploratory trials.

### 2.2 Architecture and input protocol

| Component | Locked choice | Justification for manuscript and reviewer response |
|---|---|---|
| Candidate backbones | The eleven architectures in the submitted study: VGG16, VGG19, ResNet50, ResNet101, ResNet152, InceptionV3, InceptionResNetV2, DenseNet169, DenseNet201, MobileNetV2, and Xception | These are the prespecified benchmark candidates. The revision does not add or remove a backbone after reviewing revised results. |
| Input image | One 256×256 RGB PET polar-map JPEG per patient | This matches the input representation in the published Teuho reference CNN and preserves the direct benchmark rationale.[1] |
| Preprocessing | Official Keras `preprocess_input` for the selected backbone | This is the reviewer-required correction. It enables each ImageNet checkpoint to receive its intended input representation; it is not a performance-tuning option.[3] |
| Classifier head | Retain the submitted topology: `Flatten → Dense(1024, ReLU) → Dropout(d) → Dense(512, ReLU) → Dropout(d) → Dense(256, ReLU) → Dropout(d) → Dense(1, sigmoid)`, where `d` is selected as one common dropout rate | Reviewer 1 requested trainable and total parameter counts, not a replacement head. Retaining the topology keeps the revision close to the submitted study; the submitted 0.50 dropout remains the centre candidate in the common regularisation search. The manuscript must describe the benchmark as a comparison of complete adapted architectures, not as a pure isolated-backbone comparison. |
| Parameter reporting | Record total and trainable parameter counts for every backbone-plus-head model and for each training phase | Direct Reviewer 1 request. This makes clear that `Flatten()` can produce different parameter totals across backbones. |
| Data augmentation | None | The legacy red/green channel shift and MixUp paths are removed. No clinically validated PET-polar-map augmentation policy has been adopted, so avoiding transformations is the conservative and reproducible choice. |

### 2.3 Fixed training choices

The settings below are **not** omitted from the scientific rationale. They are fixed deliberately so that the benchmark compares architectures under a common training framework rather than comparing a changing mixture of backbones, optimisers, losses, augmentations, and manual adjustments.

| Setting | Locked choice | Rationale |
|---|---|---|
| Optimiser | Adam, with common Keras default beta and epsilon values | Adam is retained from the submitted TL protocol. It is not claimed to be universally superior to SGD or AdamW. It is held constant so that the experiment compares backbones under one common optimisation family; optimiser-family comparison would require separate learning-rate and momentum schedules and would answer a different research question.[4] |
| Loss | Unweighted binary cross-entropy | The training subset contains 25 positive and 36 negative patients. This is a moderate, not extreme, imbalance. Unweighted loss allows every patient to contribute equally and does not deliberately shift the probability-learning objective toward positive cases. This is especially appropriate because Reviewer 1 requires probability calibration for downstream ensemble analysis. |
| Class weights | Disabled | Positive-class weighting would alter the loss so that a positive-patient error counts more than an equally sized negative-patient error. The legacy balanced formula would give weights of approximately 1.22 to positives and 0.85 to negatives. Without a prespecified clinical cost ratio favouring sensitivity, we should not impose such a shift. Class-imbalance correction can change calibration and does not automatically improve discrimination; the cited clinical-prediction evidence is a warning rather than direct proof for this CNN application.[5] |
| Batch size | 5, applied to all architectures | This was the submitted Methods value and is feasible for large 256×256 pretrained backbones. One common batch size avoids architecture-specific update schedules. The earlier fixed-split launcher currently uses 10; the revised protocol intentionally standardises this to 5 and documents the revision. |
| Head-training budget | Maximum 30 epochs | Retains the submitted three-phase training budget. The epoch limit is a ceiling, not a selected final epoch. |
| Fine-tuning budget | Maximum 30 epochs in Phase 2 and 30 epochs in Phase 3 | Retains the submitted maximum total of 90 epochs. |
| Early stopping | Monitor validation binary accuracy; patience 4 epochs; restore best weights | Retains the submitted rule to avoid unnecessary redesign. It must operate only on the relevant development/fixed validation subset and be logged for every run. |
| Individual-model threshold | Explicit `probability >= 0.50` | Retains the intended Version 1 operating threshold but replaces ambiguous `np.round()` code. It is not tuned on the test cohort. |
| Final stability seeds | Predeclare seeds 1–100 | Preserves the Teuho-matched repeated-run stability comparison. Seeds are not independent patient samples and are never used for run-level inferential p-values. |

### 2.4 Progressive fine-tuning: retain the idea, repair the implementation

The revised workflow retains the submitted three-phase idea:

```text
Phase 1: Frozen ImageNet backbone; train the retained classifier head.
Phase 2: Train the retained head plus a predefined terminal native backbone stage.
Phase 3: Train the retained head plus the predefined terminal two native backbone stages.
```

The legacy implementation attempts to unfreeze the second-to-last and third-to-last **individual layers** after compiling the model. That is not a comparable definition across VGG, ResNet, DenseNet, Inception, MobileNetV2, and Xception, and Keras requires recompilation after `trainable` settings are changed.[4]

The revision is therefore an implementation repair, not a depth search. Before running, the code will contain a frozen architecture registry that defines Phase 2 and Phase 3 using native high-level stage/block groupings for every backbone. For example, the terminal VGG block, the final ResNet stage, or the final DenseNet dense block will be treated as the comparable unit. The registry will be committed before training and will not be modified after observing results.

At the transition to each fine-tuning phase, the code will set trainability according to this fixed registry, instantiate a fresh Adam optimiser with the selected fine-tuning learning rate, recompile, and then continue training. For architectures containing Batch Normalization, the pretrained backbone will be called with `training=False` so that running Batch Normalization statistics are not overwritten by very small batches during fine-tuning.[4]

---

## 3. Development-only common hyperparameter search

### 3.1 Why there is a search

Reviewer 1 asks the authors to describe the hyperparameter search space and selection procedure. Neither CLAIM nor TRIPOD+AI specifies a mandatory universal grid; both instead require transparent reporting of values used, ranges searched, the search method, the data partitions, and the final model-selection rule.[6] [7]

The search is restricted to the three retained-pipeline controls that directly govern optimisation and head regularisation. It does **not** search classifier-head topology, augmentation, batch size, optimiser family, class weights, threshold, or input resolution. Searching all of those factors would turn a correction of the submitted benchmark into a large, flexible new model-development study. Broad tuning on only 92 development patients also increases the risk of selecting a configuration that wins by noise in cross-validation rather than by genuine performance.[8]

### 3.2 Candidate common protocols

| Search factor | Candidate values | Why these values are justified |
|---|---:|---|
| Head-training Adam learning rate | `3e-5`, `1e-4`, `3e-4` | The submitted protocol used `1e-4`. The lower and higher values are approximately one-third and three times that legacy centre value, providing a compact log-spaced neighbourhood rather than an unrestricted range. |
| Fine-tuning Adam learning rate | `1e-6`, `3e-6`, `1e-5` | After pretrained layers are unfrozen, Keras advises a very low learning rate because small datasets can otherwise rapidly damage pretrained features. `1e-5` is Keras’s worked-example value; the two lower values test more cautious adaptation.[4] |
| Dropout rate in the retained head | `0.30`, `0.50`, `0.60` | The submitted model used 0.50 after each of three Dense layers. The range tests weaker and moderately stronger regularisation while keeping the legacy value as the centre candidate. Values substantially above 0.60 are not included because three stacked dropout layers could over-regularise the large retained head. |

This creates **27 prespecified common protocols**:

```text
3 head learning rates × 3 fine-tuning learning rates × 3 dropout rates = 27 protocols.
```

The 27 protocols are applied identically to all 11 architectures. The development search therefore requires 27 protocols × 5 folds × 11 architectures = **1,485 training fits**, using one common fixed tuning seed. This is a development-only step. The 100-seed repetition is reserved for the final locked fixed-split analyses, not multiplied across candidate protocols.

### 3.3 Common protocol selection rule

The revised `TL_crossvalidation.py` will use only the 92 development patients. It will create one fixed, patient-level, outcome-stratified five-fold partition and apply the exact same folds to every architecture and every candidate protocol. The 46 test patients and their labels will not be loaded by this script.

For every architecture/protocol combination, the script will save one held-out probability for every development patient. These 92 out-of-fold probabilities are used to calculate that architecture’s pooled out-of-fold AUC for that protocol. For each common protocol, the study then calculates the unweighted mean of the 11 architecture-specific pooled out-of-fold AUC values. Every architecture contributes one value, so no model receives extra influence or extra tuning trials.

> **Selection rule:** choose the common protocol with the highest mean pooled out-of-fold AUC across the eleven architectures. If two protocols are numerically exactly tied at the retained reporting precision, select the lower fine-tuning learning rate; if still tied, select the lower head-training learning rate; if still tied, select dropout 0.50 because it is the submitted value. This rule is fixed before revised test results are accessed.

The selected common protocol is then frozen. The revised fixed-split script uses it unchanged for all eleven architectures and all 100 final seeds.

This answers a deliberately narrow question: **which architecture performs best under one common, development-selected transfer-learning recipe?** It does not claim to estimate each architecture’s maximum possible performance after unlimited architecture-specific tuning.

---

## 4. Data partitions, seeds, and outputs

### 4.1 Roles of the two TL scripts

| Script | Data it may use | Purpose | It must not do |
|---|---|---|---|
| `TL_crossvalidation.py` | The 92 development patients only | Evaluate the 27 common candidate protocols and produce patient-level out-of-fold probabilities for selection | Load, evaluate, rank, tune, or otherwise access the 46 test patients |
| `TL_fixedvalidation.py` | Historic 61-training, 31-validation, and 46-test patient partitions | Run each locked architecture/protocol for 100 predeclared seeds and generate final per-patient prediction outputs | Select the protocol, select a best seed, select a winning architecture, or alter settings after viewing test results |

### 4.2 Final fixed-split runs

After the common protocol is locked, each architecture is run on the historic comparison structure:

```text
61 training patients → training and class-weight-free optimisation
31 validation patients → early stopping only
46 test patients → one final probability vector per seed
100 predeclared seeds per architecture → descriptive stability distribution
```

The revised code must preserve every seed-specific probability at full precision. It must not store predictions only as a comma-separated string with an assumed order.

Each row of the secure prediction output will include:

| patient/image identifier | split | model | protocol ID | seed | observed label | predicted probability | 0.50 binary prediction |
|---|---|---|---|---:|---:|---:|---:|
| Secure local identifier | test | Xception | `HPxx` | 17 | 1 | 0.821384 | 1 |

Patient identifiers, images, labels, and clinical data remain on secure hospital infrastructure. GitHub will receive code, configuration templates, and non-sensitive schemas only.

### 4.3 Later evaluation enabled by these outputs

The patient-level files enable the later locked evaluation required by Reviewer 1. For each method, the 100 seed probabilities can be aggregated with the predeclared rule to yield one final probability per test patient. The later analysis will then calculate discrimination, calibration, bootstrap confidence intervals, and paired patient-level comparisons without falsely treating 100 seed runs as 100 clinical cohorts.[2]

The TL scripts themselves do not calculate final calibration or paired inference. Their responsibility is to save the correct locked probabilities for the statistics/evaluation layer.

---

## 5. Reviewer-response text: working draft for Comment 8

The following text is a working draft. Replace bracketed fields only after revised runs and outputs are available.

> We thank the reviewer for identifying the preprocessing issue. In the original implementation, all input images were rescaled to the 0–1 range. Because the selected ImageNet-pretrained Keras architectures have architecture-specific preprocessing requirements, we repeated the transfer-learning benchmark using the official Keras `preprocess_input` function corresponding to each backbone. All models used 256×256 RGB polar maps, matching the input representation of the published reference CNN. No image augmentation was applied.
>
> To preserve a controlled benchmark, the submitted common classifier-head topology was retained for all architectures. We now report total and trainable parameter counts for every complete model and training phase. The intended progressive three-phase fine-tuning procedure was also reimplemented using predefined architecture-specific terminal backbone stages, with recompilation after every change in layer trainability.
>
> Hyperparameter selection was conducted exclusively in the 92-patient development cohort using stratified five-fold cross-validation. The prespecified common search comprised 27 protocols formed from three head-training learning rates (`3e-5`, `1e-4`, `3e-4`), three fine-tuning learning rates (`1e-6`, `3e-6`, `1e-5`), and three dropout rates (`0.30`, `0.50`, `0.60`). For each protocol, pooled out-of-fold AUC was calculated for every architecture; the common protocol with the highest mean pooled out-of-fold AUC across the eleven architectures was selected. The 46-patient test cohort was not accessed during preprocessing verification, hyperparameter selection, architecture selection, threshold selection, or model training.
>
> Adam and binary cross-entropy were used as the common optimiser and loss framework, respectively. Because the training class imbalance was modest (25 positive and 36 negative patients) and subsequent ensemble analyses use predicted probabilities, unweighted binary cross-entropy was used. The resulting locked test predictions were retained at patient level to support calibration assessment and patient-level confidence intervals. We now provide exact data partitions, random seeds, hyperparameter values, preprocessing functions, and code.
>
> Finally, we removed unsupported claims that transfer learning reduced computational cost relative to the reference CNN or eliminated training.

---

## 6. Manuscript Methods text: working draft

This draft is intentionally a methods template rather than final text. Insert the selected protocol ID, parameter table, package versions, exact phase-stage registry, and resulting parameter counts after implementation.

> **Transfer learning.** Eleven ImageNet-pretrained Keras Application architectures were evaluated using 256×256 RGB PET polar maps. Inputs were processed using the architecture-specific Keras `preprocess_input` function associated with each ImageNet checkpoint. Each backbone was combined with the common classifier-head topology used in the submitted study: a Flatten layer, Dense layers with 1024, 512, and 256 ReLU units, one common development-selected dropout rate after each Dense layer, and a sigmoid output. Total and trainable parameter counts were recorded for each architecture and training phase.
>
> A fixed three-phase procedure was used. During Phase 1, the ImageNet backbone was frozen and the classifier head was trained for up to 30 epochs. During Phase 2, the classifier head and a predefined terminal native backbone stage were trainable for up to 30 epochs. During Phase 3, the classifier head and the terminal two predefined backbone stages were trainable for up to 30 epochs. The relevant backbone stages were specified before model fitting for every architecture. Models were recompiled after each change in trainability. Early stopping monitored validation binary accuracy with patience 4 and restoration of the best weights.
>
> Hyperparameters were selected exclusively in the 92-patient development cohort using one fixed seed and stratified five-fold cross-validation. The common prespecified grid included head-training learning rates of `3e-5`, `1e-4`, and `3e-4`; fine-tuning learning rates of `1e-6`, `3e-6`, and `1e-5`; and dropout rates of 0.30, 0.50, and 0.60. For each protocol, pooled out-of-fold AUC was calculated separately for every architecture. The protocol with the highest mean pooled out-of-fold AUC across the eleven architectures was selected and then fixed for final evaluation. Adam was used as the common optimiser and unweighted binary cross-entropy as the loss function. No data augmentation or class weighting was used. Batch size was 5 and the individual-model classification threshold was fixed at 0.50.
>
> Each locked architecture was then evaluated using the historic 61/31/46 training/validation/test split, with 100 predeclared random seeds. The 31-patient validation subset was used only for early stopping. The 46-patient test subset was not used for hyperparameter selection, model selection, threshold selection, or training. Full-precision patient-level test probabilities were retained for the locked statistical and calibration analyses.

---

## 7. Final TL code-change checklist

The following work is the complete TL-only implementation scope. It should be completed in the R1 reanalysis code after this protocol is formally approved. No raw patient data or patient-level output is to be committed to GitHub.

### 7.1 Shared changes required in both scripts

| ID | Change | Why it is needed |
|---|---|---|
| TL-01 | Replace universal `image / 255.0` with an architecture-preprocessing registry that invokes the official Keras `preprocess_input` function for the selected model. | Mandatory response to Reviewer 1 Comment 8. |
| TL-02 | Enforce 256×256 RGB input and log image size, colour channels, preprocessing function, TensorFlow/Keras version, and ImageNet-weight source. | Enables reproducibility and the Teuho-matched input justification. |
| TL-03 | Retain the legacy Flatten/Dense topology but make its common dropout rate a command-line/configuration parameter. | The head topology is retained; dropout is a prespecified development-search factor. |
| TL-04 | Replace the hard-coded optimiser learning rate with explicit `--head_learning_rate` and `--fine_tune_learning_rate` arguments. | Enables the prespecified 27-protocol search and correct phase-specific rates. |
| TL-05 | Replace the current `--det`/hard-coded seed 43 mechanism with `--seed`, a unified seed-setting function, and a logged seed value. | Direct response to reproducibility and seed-reporting requests. |
| TL-06 | Disable class weights by default and log `class_weights=false`; remove the launcher flag that enables them. | Implements the agreed unweighted BCE policy. |
| TL-07 | Use an explicit 0.50 threshold comparison, not `np.round()`, for binary metrics. Save both the probability and thresholded class. | Makes the decision rule unambiguous and reproducible. |
| TL-08 | Build a parameter-report function that writes total, trainable, and non-trainable parameter counts at every phase. | Direct Comment 8 requirement. |
| TL-09 | Build a fixed architecture fine-tuning registry. It must define Phase 2 and Phase 3 terminal native stage/block layers for every one of the eleven backbones. | Replaces incomparable “second-to-last layer” logic while retaining the three-phase concept. |
| TL-10 | After every trainability change: apply the registry, keep Batch Normalization in inference mode, create a fresh Adam optimiser with the selected phase learning rate, and recompile before fitting. | Makes fine-tuning technically valid under Keras guidance.[4] |
| TL-11 | Log a full run manifest: model, protocol ID, all hyperparameters, split/fold identity, seed, package versions, epoch history, early-stopping epoch, phase parameter counts, and code commit. | Provides the exact reproducibility record requested by Reviewer 1. |

### 7.2 `TL_crossvalidation.py`: development-only selection

| ID | Change | Why it is needed |
|---|---|---|
| CV-01 | Replace `KFold` with one fixed `StratifiedKFold` partition over the 92 development labels, saved securely as a fold manifest. | Preserves comparable event distributions across folds and provides exact reproducible partitions. |
| CV-02 | Remove any test-data path, test-label loading, and test evaluation from the CV script. | Ensures that protocol selection uses development data only, as required by Comment 3. |
| CV-03 | Implement the 27 common protocol grid exactly as specified in this document. Each architecture receives every protocol, each uses the same five folds, and each uses the same fixed tuning seed. | Produces a controlled, equal-budget common-protocol search. |
| CV-04 | Save a row per held-out patient per fold with secure patient/image ID, label, architecture, protocol ID, fold, seed, probability, and thresholded prediction. | Creates valid out-of-fold records instead of comma-separated vectors without identity. |
| CV-05 | Produce a configuration summary with pooled OOF AUC per architecture and mean pooled OOF AUC across all eleven architectures. Apply the predeclared tie rule and write a read-only selected-protocol manifest. | Locks the development-selected common protocol before fixed-split testing. |
| CV-06 | Save all 27 protocol results, not only the winner. | Allows transparent supplement/reporting and avoids appearance of cherry-picking. |

### 7.3 `TL_fixedvalidation.py`: final historic benchmark runs

| ID | Change | Why it is needed |
|---|---|---|
| FV-01 | Read the selected-protocol manifest rather than accept unverified ad hoc settings. | Prevents silent post-selection changes between CV and final testing. |
| FV-02 | Read a secure explicit 61/31/46 split manifest; verify counts, labels, no overlap, and filename/label alignment before fitting. | Documents the historic benchmark split and prevents accidental split drift. |
| FV-03 | Iterate exactly over predeclared seeds 1–100 for each of the eleven locked architectures. | Retains stability comparison while making the seed list auditable. |
| FV-04 | Atomically store one full-precision patient-level **test** prediction per completed model/seed with secure patient ID, protocol ID, seed, observed label, probability, and binary prediction in one secure SQLite results database. | Enables later seed aggregation, calibration, bootstrap CIs, paired tests, and EL construction without ambiguous Excel strings or thousands of files. |
| FV-05 | Store run metrics, phase parameters, test predictions, and completion status in the same resumable database; reject incompatible protocol/split/code signatures and permit only structurally complete runs to be skipped on resume. | Corrects the legacy overwriting risk while retaining one compact working-results file. |
| FV-06 | Export one validated, human-readable Excel workbook only after all expected runs are complete; do not calculate Kruskal–Wallis, Mann–Whitney, best-seed, or test-selected winner outputs. | Retains practical inspectability while addressing Reviewer 1 Comments 4 and 5. |

### 7.4 Launcher and configuration changes

| ID | Change | Why it is needed |
|---|---|---|
| RUN-01 | Replace ad hoc bash values with a single protocol configuration file generated by the CV selection stage. | Ensures both scripts use the same locked common settings. |
| RUN-02 | Set input size 256, batch size 5, class weights off, no augmentation, and explicit seed list in the final launcher. | Implements locked protocol consistently. |
| RUN-03 | Maintain a non-sensitive configuration template in GitHub; keep secure split manifests, images, IDs, labels, model weights, and patient-level predictions outside GitHub. | Protects patient data while permitting code reproducibility. |

---

## 8. Items intentionally outside this TL implementation

| Item | Where it belongs |
|---|---|
| ICA/FFR reference-standard wording, FFR counts, multivessel handling, and MBF role | Comment 1 clinical clarification and manuscript revision |
| Conventional stress-MBF threshold, logistic, and clinical baselines | Comment 2 baseline-analysis code |
| Ensemble pool/rule/weight/threshold selection and complete Borda removal | Dedicated EL workflow; it consumes CV outputs but must not modify TL selection after test review |
| One locked prediction per test patient, calibration metrics/plot, bootstrap confidence intervals, paired DeLong and McNemar analyses | Final statistics/evaluation workflow |
| Clinical-reader documentation and comparison wording | Comment 6 response/manuscript work |
| Sensitivity/specificity interpretation of the Max Rule | Comment 7 response/manuscript work |

---

## 9. Implementation gate before GPU runs

Before starting the 1,485 development fits or the 11 × 100 final fixed-split runs, confirm all of the following:

1. The clinical endpoint wording and labels are frozen for this reanalysis.
2. The secure patient-order mapping and 61/31/46 split are verified.
3. The architecture-specific preprocessing registry is unit-tested using sample images.
4. The Phase 2 and Phase 3 architecture registry has been reviewed for all eleven backbones.
5. The 27-protocol grid, selection metric, tie rule, five folds, and tuning seed are written to version-controlled configuration.
6. Test labels are inaccessible to the CV-selection workflow.
7. The final output schema includes patient ID, seed, protocol ID, probability, and label.
8. The code is committed and tagged before computation begins.

Only after these gates are satisfied should the revised GPU experiments begin.

---

## References

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8857225/ "Classification of ischemia from myocardial polar maps in 15O-water PET"

[2]: https://www.bmj.com/content/385/bmj-2023-078378 "TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods"

[3]: https://keras.io/api/applications/ "Keras Applications API documentation"

[4]: https://keras.io/guides/transfer_learning/ "Keras guide to transfer learning and fine-tuning"

[5]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9382395/ "The harm of class imbalance corrections for risk prediction models"

[6]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11304031/ "Checklist for Artificial Intelligence in Medical Imaging: CLAIM 2024 Update"

[7]: https://www.bmj.com/content/385/bmj-2023-078378 "TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods"

[8]: https://jmlr.csail.mit.edu/papers/v11/cawley10a.html "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation"

[9]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9007400/ "Transfer learning for medical image classification: a literature review"

[10]: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0262838 "Deep learning model calibration for improving performance in class-imbalanced medical image classification tasks"
