# ICH-Assitant

**ICH Assistant** is an automated pipeline specifically designed for patients suffering from intracranial hemorrhage (ICH). This pipeline integrates deep learning for the automatic detection, segmentation, and volume measurement of ICH, lesion location registration, and a clinical decision-support system based on a large language model (GPT-4o-mini). The goal of this pipeline is to enhance therapeutic outcomes and improve the quality of life for patients by supporting clinical decision-making in head CT analysis.

## Project Overview
This repository contains the code, related files, and instructions for the **ICH Assistant** pipeline, which includes the following components:

### 1. Detection
The detection module resamples images to a slice thickness of 5mm and converts CT scans into three-channel images to highlight different tissue types. The detection model uses a Swin Transformer for intra-slice feature extraction and a Sequence Transformer for inter-slice feature integration.

- **Model Architecture**: Swin Transformer + Sequence Transformer
- **Code Location**: `./Detection`
- **Additional Reference**: The detection model is based on the work available at [Effective Transformer-based Solution for RSNA Intracranial Hemorrhage Detection](https://github.com/PaddlePaddle/Research/tree/master/CV/Effective%20Transformer-based%20Solution%20for%20RSNA%20Intracranial%20Hemorrhage%20Detection).

### 2. Segmentation
The segmentation module uses the nnU-NetV2 model to automatically segment different types of hemorrhage, including intraparenchymal hemorrhage (IPH), intraventricular hemorrhage (IVH), subarachnoid hemorrhage (SAH), subdural hematoma (SDH), and epidural hematoma (EDH). Additionally, it identifies perihematomal edema (PHE).

- **Model Architecture**: nnU-NetV2
- **Code Location**: `./Segmentation`
- **Data**: The segmentation folder includes masks segmented by doctors using the CQ500 dataset.
- **Additional Reference**: The segmentation model is based on the [nnUNetV2 project](https://github.com/MIC-DKFZ/nnUNet).

### 3. Registration
The registration module automatically determines the location of the lesions, including IPH, IVH, SAH, SDH, EDH, and PHE.

- **Output**: Lesion locations in relation to a standard brain template.
- **Code Location**: `./Registration`
- **Resources**: Includes standard brain templates and detailed documentation.



### 5. Clinical Decision Support
The Clinical Decision Support module processes the output from the detection, segmentation, registration, and hydrocephalus and midline shift modules, generating case-specific recommendations for further examination and treatment.

- **Process**: The module exports ICH evaluation results as JSON files, and uses a large language model (GPT-4o-2024-11-20) to provide clinical guidance based on the following guidelines and clinical tirals:
  - Greenberg, Steven M et al. 2022 Guideline for the Management of Patients With Spontaneous Intracerebral Hemorrhage.
  - Hawryluk, Gregory W J et al. 2020 Update of the Decompressive Craniectomy Recommendations.
  - Hoh, Brian L et al. 2023 Guideline for the Management of Patients With Aneurysmal Subarachnoid Hemorrhage.
  - Li G, Lin Y et al. Intensive Ambulance-Delivered Blood-Pressure Reduction in Hyperacute Stroke. 
  - Pradilla G, Ratcliff JJ et al. Trial of Early Minimally Invasive Removal of Intracerebral Hemorrhage. 
  - Ma L, Hu X et al. The third Intensive Care Bundle with Blood Pressure Reduction in Acute Cerebral Haemorrhage Trial (INTERACT3): an international, stepped wedge cluster randomised controlled trial.
  - Connolly SJ, Sharma M et al, Andexanet for Factor Xa Inhibitor-Associated Acute Intracerebral Hemorrhage
  - Beck J, Fung C et al,. Decompressive craniectomy plus best medical treatment versus best medical treatment alone for spontaneous severe deep supratentorial intracerebral haemorrhage: a randomised controlled clinical trial
- **Code Location**: `./Clinical Decision Support`
