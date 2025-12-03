# MRI Data Sources

## Where to Find MRI Images

This document provides detailed information about publicly available MRI datasets suitable for tissue segmentation analysis.

## Recommended Datasets

### 1. MICCAI Brain Tumor Segmentation Challenge (BraTS)

**Website:** http://braintumorsegmentation.org/

**Description:**
- Annual challenge with brain tumor MRI data
- Multi-institutional, multi-scanner data
- High-quality annotations by expert neuroradiologists

**Data Contents:**
- T1-weighted (T1)
- T1-weighted with gadolinium contrast (T1-Gd)
- T2-weighted (T2)
- Fluid Attenuated Inversion Recovery (FLAIR)
- Ground truth segmentations (tumor regions)

**Format:** NIfTI (.nii.gz)

**Access:** Registration required

**How to Download:**
1. Visit the BraTS website
2. Register for an account
3. Navigate to the data download section
4. Download training/validation sets
5. Place files in `data/raw/`

---

### 2. OASIS (Open Access Series of Imaging Studies)

**Website:** https://www.oasis-brains.org/

**Description:**
- Cross-sectional and longitudinal brain MRI datasets
- Normal aging and Alzheimer's disease subjects
- Well-characterized with demographic and clinical data

**Available Projects:**
- **OASIS-1**: Cross-sectional (ages 18-96)
- **OASIS-2**: Longitudinal (multiple timepoints)
- **OASIS-3**: Large-scale longitudinal

**Format:** NIfTI

**Access:** Free with account registration

**How to Download:**
1. Create account on OASIS website
2. Browse available datasets
3. Download individual scans or bulk data
4. Extract and place in `data/raw/`

---

### 3. Human Connectome Project (HCP)

**Website:** http://www.humanconnectomeproject.org/

**Description:**
- High-resolution brain imaging
- Structural and functional MRI
- Excellent for detailed tissue analysis

**Data Contents:**
- T1-weighted structural MRI
- T2-weighted structural MRI
- Diffusion MRI
- Resting-state and task fMRI

**Format:** NIfTI

**Access:** Free with data use agreement

**How to Download:**
1. Register at ConnectomeDB
2. Accept data use terms
3. Use Amazon S3 or Aspera for downloads
4. Place structural scans in `data/raw/`

---

### 4. ADNI (Alzheimer's Disease Neuroimaging Initiative)

**Website:** http://adni.loni.usc.edu/

**Description:**
- Longitudinal brain MRI for Alzheimer's research
- Multiple scanning protocols
- Rich clinical metadata

**Data Contents:**
- T1-weighted MRI
- T2-weighted MRI
- FLAIR
- PET imaging
- Clinical assessments

**Format:** Various (DICOM, NIfTI available)

**Access:** Registration and approval required

**How to Download:**
1. Register at ADNI website
2. Complete Data Use Agreement
3. Search and download via LONI IDA
4. Convert DICOM to NIfTI if needed
5. Place in `data/raw/`

---

### 5. IXI Dataset

**Website:** https://brain-development.org/ixi-dataset/

**Description:**
- Nearly 600 MR images from healthy subjects
- Multiple imaging modalities
- Ages 20-86

**Data Contents:**
- T1-weighted
- T2-weighted
- PD-weighted
- MRA
- Diffusion MRI

**Format:** NIfTI

**Access:** Freely available (cite if used)

**How to Download:**
1. Visit IXI website
2. Direct download from Imperial College London
3. No registration required
4. Place in `data/raw/`

---

## File Organization

Organize downloaded files in the following structure:

```
data/
└── raw/
    ├── subject_001/
    │   ├── T1.nii.gz
    │   ├── T2.nii.gz
    │   └── FLAIR.nii.gz
    ├── subject_002/
    │   └── T1.nii.gz
    └── ...
```

## Data Preprocessing Steps

Before using the data with our pipeline:

1. **Check format**: Ensure files are NIfTI (.nii or .nii.gz)
2. **Verify orientation**: Check if reorientation is needed
3. **Skull stripping** (optional): Remove non-brain tissue
4. **Bias field correction** (optional): Correct intensity inhomogeneity

## Tools for Data Conversion

If you receive DICOM files:

### Using dcm2niix (Recommended)
```bash
# Install
sudo apt-get install dcm2niix  # Linux
brew install dcm2niix          # macOS

# Convert
dcm2niix -o data/raw/ -f subject_001 path/to/dicom/folder/
```

### Using Python (nibabel + pydicom)
```python
import nibabel as nib
import pydicom
from pydicom.data import get_testdata_files

# Load DICOM series and convert
# (Implementation depends on your DICOM structure)
```

## Sample Data for Testing

If you don't have access to real MRI data yet, the code includes synthetic data generation:

```python
from src.utils import create_synthetic_mri_data

# Create synthetic brain-like MRI
image, ground_truth = create_synthetic_mri_data(
    shape=(256, 256),
    n_tissues=3,
    noise_level=0.1
)
```

## Data Size Considerations

- **Single MRI volume**: ~1-50 MB (compressed NIfTI)
- **Complete BraTS dataset**: ~20-30 GB
- **OASIS-3**: ~1 TB (full dataset)
- **Recommended for this project**: Start with 10-50 scans

## Citation Requirements

If you use these datasets, please cite appropriately:

**BraTS:**
```
Menze et al., "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)",
IEEE Transactions on Medical Imaging, 2015
```

**OASIS:**
```
Marcus et al., "Open Access Series of Imaging Studies (OASIS)",
Journal of Cognitive Neuroscience, 2007
```

**HCP:**
```
Van Essen et al., "The WU-Minn Human Connectome Project",
NeuroImage, 2013
```

## Additional Resources

- **NiBabel Documentation**: https://nipy.org/nibabel/
- **FSL (FMRIB Software Library)**: https://fsl.fmrib.ox.ac.uk/
- **FreeSurfer**: https://surfer.nmr.mgh.harvard.edu/
- **ANTs**: http://stnava.github.io/ANTs/

## Troubleshooting

### Common Issues:

1. **File format errors**: Ensure files are valid NIfTI
2. **Memory errors**: Start with smaller 2D slices
3. **Different orientations**: Use nibabel to standardize
4. **Missing data**: Check download completed successfully

### Getting Help:

- Check dataset documentation
- Review NiBabel tutorials
- See project README.md for code examples
- Open issue on GitHub repository

## Next Steps

After downloading data:

1. Place files in `data/raw/`
2. Open `notebooks/tissue_segmentation_analysis.ipynb`
3. Modify data loading cell to point to your files
4. Run the analysis pipeline

Happy analyzing! 🧠
