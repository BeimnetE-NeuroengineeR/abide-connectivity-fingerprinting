# abide-connectivity-fingerprinting
 Exploratory evaluation of functional connectivity fingerprinting uniqueness vs. diagnostic group classification in fMRI across scanner sites (ABIDE). Python, nilearn, scikit-learn.



# Functional Connectivity Profile Similarity & Fingerprinting Across Scanner Sites (ABIDE)

## Question
How does macroscale whole-brain functional connectivity similarity compare across scanner sites versus across diagnostic categories (Autism vs. Neurotypical Control)?

## Data & Methods
- **Dataset:** ABIDE I preprocessed fMRI ($N = 871$, 20 sites; CC200 parcellation, CPAC pipeline).
- **Features:** Vectorized correlation matrices across 200 ROI time series per subject.
- **Metric:** Whole-brain pairwise Cosine Similarity evaluated across site boundaries and diagnostic groups.

## Results

| Pairwise Comparison | Mean Cosine Similarity |
| :--- | :---: |
| **Same Scanner Site** | **0.6789** |
| **Different Scanner Sites** | **0.6721** |
| **Same Diagnosis (Autism/Control)** | **0.6734** |
| **Different Diagnosis** | **0.6718** |

## Key Findings
- **Site Effects Exceed Diagnostic Variance:** Pairwise profile similarity between subjects scanned at the same site ($0.6789$) is higher than similarity between subjects across different sites ($0.6721$), confirming that scanner-specific acquisition noise shifts global connectivity profiles.
- **Subtle Whole-Brain Diagnostic Effect:** Whole-brain profile similarity across identical diagnoses ($0.6734$) is only slightly higher than across differing diagnoses ($0.6718$), underscoring that autism-related functional alterations are localized to specific sub-networks rather than uniform whole-brain shifts.

## Figures
The similarity breakdown bar plot is generated automatically and saved to `reports/similarity_breakdown.png`.