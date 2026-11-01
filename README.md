# MultiMedia-Authentication
# Multimedia Authentication using Multi-Feature Fusion

Deepfake **face-video detection** that looks at each face in two ways, as an ordinary RGB image (*spatial view*) and as Haar-wavelet high-frequency detail (*frequency view*), and mixes the two with a small **adaptive gate** that outputs one weight alpha per video.

B.Tech minor project, Department of Computer Science & Engineering, MNNIT Allahabad, Prayagraj (November 2026).
Inspired by Wang et al., *Deepfake Detection Based on the Adaptive Fusion of Spatial-Frequency Features*, Int. J. Intelligent Systems, 2024 (see `docs/`). This is a **reduced re-interpretation**, not a re-implementation.

## Demo video
[Watch the Gradio web-app demo](https://drive.google.com/file/d/1UQFs4Lyz_7q3jytFVDqS5oXxYwCsVcw-/view?usp=sharing)

## How it works
```
Video -> 8 evenly spaced frames -> face crop (MediaPipe BlazeFace, 1.4x box, 224x224)
      -> Spatial branch  : ResNet-18 (ImageNet) on RGB          -> 512-d vector s
      -> Frequency branch: Haar LH/HL/HH bands -> ResNet-18      -> 512-d vector f
      -> Gate: alpha = sigmoid(MLP([s ; f]))   (1024 -> 128 -> 1)
      -> fused = alpha*s + (1-alpha)*f -> classifier 512 -> 256 -> 2 -> REAL / FAKE
```
Key choices: source-wise train/val/test split (no identity leakage), class-weighted loss, modality dropout (20%), zero-initialised gate (alpha starts at 0.5), checkpoint chosen by validation ROC-AUC.

| Item | Base paper (Wang et al.) | This project |
|---|---|---|
| Backbone | ResNet-34 | ResNet-18 |
| Frequency input | Daubechies wavelet + residual filters | Haar high-frequency bands |
| Fusion | Gated convolution on feature maps | One scalar gate per video |
| Decision unit | Single frame | Video (8 frames averaged) |

## Results (FaceForensics++ C23, 888 unseen test videos)
| Model | ROC-AUC | Accuracy | REAL recall | FAKE recall | Bal. acc. |
|---|---|---|---|---|---|
| Spatial ResNet-18 | 0.988 | 96.2% | 92.6% | 96.9% | 94.7% |
| Frequency (Haar) ResNet-18 | 0.969 | 92.6% | 77.7% | 95.5% | 86.6% |
| Late fusion | 0.990 | 96.7% | 92.6% | 97.6% | 95.1% |
| **Gated fusion (ours)** | **0.989** | **96.0%** | **91.2%** | **96.9%** | **94.0%** |

Honest reading: the gated model is **statistically tied with the spatial branch** (paired bootstrap). The gate does respond to the input (mean alpha 0.58 for real vs 0.73 for fake videos), but forcing alpha = 1 barely changes the score.

| ROC curves | Gate alpha by class |
|---|---|
| ![ROC](results/roc_v2.png) | ![Gate](results/gate_alpha_v2.png) |
| **Per-method accuracy** | **Confusion matrix (gated)** |
| ![Per method](results/per_method_v2.png) | ![Confusion](results/confusion_gated_v2.png) |

### Cross-dataset (no retraining, threshold 0.5, small samples)
| Evaluation set | Videos | ROC-AUC | REAL recall | FAKE recall |
|---|---|---|---|---|
| FF++ C23 test | 888 | 0.989 | 91.2% | 96.9% |
| DFD sample | 174 | 0.763 | 29.8% | 93.3% |
| Celeb-DF v2 sample | 199 | 0.585 | 29.3% | 82.0% |

Performance drops strongly outside FaceForensics++. A threshold tuned on a small target sample adds only about 4 points of balanced accuracy, within the measurement uncertainty. We make **no claim of strong generalisation**.

## Repository structure
```
notebooks/
  01_train_and_evaluate_ffpp.ipynb   face extraction, branch + gated training, FF++ evaluation, plots
  02_cross_dataset_dfd.ipynb         DFD sample: gated vs spatial vs frequency vs late fusion, forced-alpha tests
  03_cross_dataset_celebdf_v2.ipynb  Celeb-DF v2 official test list: standalone branches vs gated
  04_threshold_calibration.ipynb     threshold chosen on a small target sample, applied to the test sample
results/                             ROC, per-method accuracy, gate histogram, confusion matrix
docs/                                project report, literature survey and plan, base paper
slides/                              project presentation (.pptx)
requirements.txt
```

## Datasets (not included)
| Dataset | Used for | Source |
|---|---|---|
| FaceForensics++ C23 | training, validation, test | Kaggle `xdxd003/ff-c23` (original: [ondyari/FaceForensics](https://github.com/ondyari/FaceForensics), check its Terms of Use) |
| DFD | cross-dataset sample | Kaggle `sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset` |
| Celeb-DF v2 | cross-dataset sample | Kaggle `reubensuju/celeb-df-v2` |

Please respect each dataset's license and terms of use. Do not commit videos to this repository.

## How to run (Google Colab, T4 GPU)
1. Mount Google Drive; the notebooks use the folder `MyDrive/Multimedia_Authentication/` (training outputs go in `v2/`).
2. Run `01_train_and_evaluate_ffpp.ipynb` top to bottom. It installs `mediapipe`, downloads the BlazeFace detector and FF++ via `kagglehub` (a Kaggle login may be required), extracts faces, trains both branches (10 epochs) and the gated model (6 epochs), and saves checkpoints (`spatial_resnet18_v2_best.pth`, `frequency_resnet18_v2_best.pth`, `gated_v2_best.pth`) and plots. A full run takes about 6 hours.
3. Run notebooks `02` and `03` for the cross-dataset tests (they load the checkpoints from step 2), then `04` for threshold calibration.

Local use: `pip install -r requirements.txt` and adjust the Drive paths at the top of each notebook. The random seed is fixed to 42.

## Gradio web app
The report describes a Gradio app that samples 8 frames, crops the face with the training pipeline, and shows the verdict, P(fake), the alpha split and the face crops (threshold 0.5). **The app source and the trained weights are not included in this repository yet**: add `app.py` and publish the weights (for example as a GitHub Release asset) before sharing the repo.

## Limitations
- Weak cross-dataset generalisation (small samples, causes not diagnosed).
- Real recall (91%) is below fake recall (97%); only C23 compression tested.
- Splits group by the first ID in `A_B.mp4`, so some identity overlap is possible versus the official splits.
- Simplified model (ResNet-18, Haar only); the gate is not shown to beat the spatial branch.

## Future work
Full DFD and Celeb-DF evaluation, official FF++ splits and C40, Daubechies wavelets and a residual-filter branch, gated convolution on feature maps, probability calibration, and testing the app on phone-recorded video.

## Team
Kirti Srivastava (20233163), Komal Singh (20233164), Divya Anand (20233120), Bodke Rushikesh Maroti (20233104)

**Guide:** Dr. Anuja Dixit, Dept. of CSE, MNNIT Allahabad

## References
- F. Wang et al., *Deepfake Detection Based on the Adaptive Fusion of Spatial-Frequency Features*, Int. J. Intelligent Systems, 2024, Article 7578036.
- A. Rossler et al., *FaceForensics++: Learning to Detect Manipulated Facial Images*, ICCV 2019.
- Y. Li et al., *Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics*, CVPR 2020.
- K. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.

## License
No license has been chosen yet. Add one (for example MIT) before publishing; the datasets and the base paper keep their own licenses.
