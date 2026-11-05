# Cervical transformation-zone type classification

Classifying cervical images into transformation zone **Type 1, 2 or 3**
with an ensemble of three EfficientNet models, calibrated probabilities,
and a Streamlit demo.

> Research project. Not a medical device and not for clinical use.

**Live demo:** (add the Streamlit link here after deployment)

## Why this problem
## Why this problem

India has about 127,500 new cervical cancer cases and about 79,900 deaths a
year (WHO/IARC, 2022), roughly one in four of the world's deaths from this
disease. Screening is limited: in a 2015-16 national survey, only 29.8% of
women aged 30-49 had ever been screened, and the figure was lower in rural
areas (27.5%) than in urban areas (34.0%). Many rural health centres lack
colposcopy and trained staff.

Transformation-zone type matters for treatment. WHO says ablation is suitable
only when the zone is fully visible and is Type 1. Type 2 or Type 3 should be
referred for colposcopy. Recognising the type from an image takes training.

This project is a research prototype that tests whether a deep learning model
can predict the type from a cervical image. It is not a diagnostic tool and
has not been clinically validated.


## Data
- Kaggle *Intel & MobileODT Cervical Cancer Screening* images:
  1,480 usable training images, split 1,036 / 222 / 222 (train / val / test).
- 6,489 extra cleaned images added to training (7,525 training images in total).
- The split is by image, not by patient.

## Method
1. Transfer learning with EfficientNet B0 (256 px), B0 (384 px) and B2 (288 px).
2. Pad to square, resize, class-weighted loss with label smoothing,
   early stopping on validation log loss.
3. Average the three softmax outputs.
4. Temperature scaling (T = 0.803) to calibrate the probabilities.

## Results (held-out test set, n = 222, 95% bootstrap CIs)
| Metric | Value |
|---|---|
| Accuracy | 0.752 (0.694-0.806) |
| Log loss | 0.560 (0.473-0.659) |
| Macro AUC | 0.895 (0.855-0.931) |
| Type 1 recall | 0.605 |
| Quadratic kappa | 0.692 |

- 55 errors, 53 of them one step off (Type 1 vs 2 or Type 2 vs 3).
- Without the extra data, validation accuracy was 0.599; with it, 0.694.
- Keeping only the 60% most confident predictions gives 0.917 accuracy.

## Second check: stage-1 test images (n = 512)
Accuracy 0.781 (0.746-0.815), log loss 0.577, macro AUC 0.896.
These are images from the same competition, not an external dataset.
Accuracy was 0.927 on images close to training images and 0.744 on images
far from them, so the 0.744 figure is the more conservative estimate.

## Limitations
- Image-level split, so near-duplicate images may inflate scores.
- Labels in this dataset are noisy.
- Type 1 is the hardest class (recall 0.605 on test).
- Single dataset, no external validation from another clinic or device.

## Repository layout
```
app.py            Streamlit demo
predict.py        command-line prediction script
weights/          trained model weights and final_pipeline.json
notebooks/        01_main_pipeline, 02_stage1_check_and_error_analysis
results/          result tables and figures
```

## Run it yourself
```
pip install -r requirements.txt
streamlit run app.py
```

## Author
Prajakta More, Electronics & Telecommunication Engineering,
MIT Academy of Engineering, Pune.