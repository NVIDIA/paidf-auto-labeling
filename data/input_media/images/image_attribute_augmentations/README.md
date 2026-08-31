# Multi-view identity images

The multi-view visual attribute search cookbook needs several camera views of
one person in a single directory. Download RSTPReid, then copy one identity
into this folder.

## Get the images

1. Download RSTPReid from
   [RSTPReid-Dataset](https://github.com/NjtechCVLab/RSTPReid-Dataset)
   ([Google Drive archive](https://drive.google.com/file/d/1HTeDZUVrZr6nL56ZlkYBNqjSWh3IGV2X/view?usp=sharing)).
2. Unpack the archive and copy RSTPReid person `0000` here as
   `00000_RSTP/`. The folder name is the five-digit identity plus `_RSTP`
   (RSTPReid `0001` → `00001_RSTP`).

RSTPReid contains 20,505 images of 4,101 people from 15 cameras. Each person
has five images from different cameras. See the upstream README for the
train/val/test split.

## What the cookbook expects

```text
data/input_media/images/image_attribute_augmentations/00000_RSTP/
  0000_c5_0022.jpg
  0000_c7_0015.jpg
  0000_c14_0032.jpg
```

The example config uses that same folder:

- group directory:
  `data/input_media/images/image_attribute_augmentations/00000_RSTP`
- representative `media_path` (one file in that directory, not the
  directory itself):
  `.../00000_RSTP/0000_c14_0032.jpg`

Copy `pipeline_image_multiview_pas.yaml` to a gitignored `*.local.yaml` and
change both `--image-group-dir` entries and `media_path` together if you use
a different identity.
