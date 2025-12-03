# Data Directory

## Purpose
This folder contains MRI imaging data for the tissue segmentation project.

## Structure

### `raw/`
Place your raw MRI NIfTI files here.

**Supported formats:**
- `.nii` - Uncompressed NIfTI
- `.nii.gz` - Compressed NIfTI (recommended)

**Recommended organization:**
```
raw/
├── subject_001/
│   ├── T1.nii.gz
│   ├── T2.nii.gz
│   └── FLAIR.nii.gz
├── subject_002/
│   └── T1.nii.gz
└── ...
```

### `processed/`
Processed data outputs will be saved here automatically by the pipeline.

## Where to Get Data

See `../docs/DATA_SOURCES.md` for detailed information about:
- MICCAI BraTS dataset
- OASIS database
- Human Connectome Project
- ADNI
- Other public MRI datasets

## Quick Start

1. Download MRI data from one of the sources listed in DATA_SOURCES.md
2. Place NIfTI files in the `raw/` subdirectory
3. Run the analysis pipeline using the Jupyter notebook or Python scripts

## Example

```python
from src.data_loader import MRIDataLoader

# Initialize loader
loader = MRIDataLoader('data/raw')

# Load a scan
volume = loader.load_nifti('data/raw/subject_001/T1.nii.gz')

# Extract a 2D slice
slice_2d = loader.extract_2d_slice(volume, slice_idx=100, axis=2)
```

## Note

**Do not commit large MRI files to Git!**

The `.gitignore` file is configured to exclude:
- `*.nii`
- `*.nii.gz`
- Large binary files

If you need to share data, use:
- External storage (Google Drive, Dropbox)
- Institutional data repositories
- AWS S3 or similar cloud storage
