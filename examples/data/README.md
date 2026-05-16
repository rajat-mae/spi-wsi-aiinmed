# Example data

Large image data are not stored in this repository. Download the public example data from Google Drive:

https://drive.google.com/file/d/1Hz-rnqrIV9KZvy3lxO_R5Awv2GaB6OKM/view

After downloading, place files like this:

```text
examples/data/
  hallmark_patient_scores.csv
  histology_images/
    SAMPLE001.tif
  sample001/
    patches/
      AAACAAGTATCTCCCA-1.png
    hallmark_pathways.csv
```

The spatial inference workflow expects the score CSV to have pathway names as rows and patient/sample IDs as columns.
