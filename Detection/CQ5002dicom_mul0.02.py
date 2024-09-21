

import os
import uuid
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import pydicom
from pydicom.dataset import FileDataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
import warnings
from contextlib import contextmanager
import sys
import io

logging.basicConfig(filename='dicom_conversion.log', level=logging.ERROR,
                    format='%(asctime)s %(levelname)s:%(message)s')

warnings.filterwarnings("ignore")

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def read_dicom_series(directory):
    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(directory)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        return image
    except Exception as e:
        logging.error(f"Error reading DICOM series from {directory}: {e}")
        return None

def resample_image(image, new_spacing=[1.0, 1.0, 1.0]):
    try:
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        new_size = [
            int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
            int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
            int(round(original_size[2] * (original_spacing[2] / new_spacing[2])))
        ]
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(new_spacing)
        resample.SetSize(new_size)
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(image.GetPixelIDValue())
        resample.SetInterpolator(sitk.sitkLinear)
        resampled_image = resample.Execute(image)
        return resampled_image
    except Exception as e:
        logging.error(f"Error resampling image: {e}")
        return None

def save_as_nii(image, output_path):
    try:
        sitk.WriteImage(image, output_path)
    except Exception as e:
        logging.error(f"Error saving NIfTI image to {output_path}: {e}")

def adjust_affine_to_dicom_standard(affine):
    RAS_to_LPS = np.array([
        [-1,  0,  0, 0],
        [ 0, -1,  0, 0],
        [ 0,  0,  1, 0],
        [ 0,  0,  0, 1]
    ])
    transformed_affine = np.dot(RAS_to_LPS, affine)
    return transformed_affine

def rotate_image_90_counterclockwise(image_data):
    return np.rot90(image_data, k=1)

def convert_nii_to_dicom(nii_file_path, dicom_template_path, output_dir, folder_name):
    try:
        nii_image = nib.load(nii_file_path)
        nii_data = nii_image.get_fdata().astype(np.int16)
        nii_header = nii_image.header
        pixdim = nii_header['pixdim']
        pixel_spacing = [str(pixdim[1]), str(pixdim[2])]
        slice_thickness = pixdim[3]

        nii_affine = nii_image.affine
        dicom_affine = adjust_affine_to_dicom_standard(nii_affine)
        direction_cosines = dicom_affine[:3, :3]
        origin = dicom_affine[:3, 3]
        spacing = np.linalg.norm(direction_cosines, axis=0)
        direction_cosines = direction_cosines / spacing

        dicom_template = pydicom.dcmread(dicom_template_path)
        if 'PixelData' in dicom_template and dicom_template.file_meta.TransferSyntaxUID.is_compressed:
            dicom_template.decompress()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        num_slices = nii_data.shape[2]

        for slice_idx in range(num_slices):
            slice_data = nii_data[:, :, slice_idx]
            slice_data = rotate_image_90_counterclockwise(slice_data)
            slice_data = np.flipud(slice_data)

            dicom_slice = FileDataset(None, {}, file_meta=dicom_template.file_meta, preamble=b"\0" * 128)
            dicom_slice.update(dicom_template)
            dicom_slice.PatientName = folder_name
            dicom_slice.PatientID = folder_name
            dicom_slice.SeriesInstanceUID = folder_name
            dicom_slice.StudyInstanceUID = folder_name
            dicom_slice.StudyDescription = folder_name
            dicom_slice.SeriesDescription = folder_name
            dicom_slice.Rows, dicom_slice.Columns = slice_data.shape
            dicom_slice.SamplesPerPixel = 1
            dicom_slice.PhotometricInterpretation = "MONOCHROME2"
            dicom_slice.PixelRepresentation = 1
            dicom_slice.BitsStored = 16
            dicom_slice.BitsAllocated = 16
            dicom_slice.HighBit = 15
            dicom_slice.PixelData = slice_data.tobytes()
            dicom_slice.RescaleIntercept = 0
            dicom_slice.RescaleSlope = 1
            dicom_slice.WindowCenter = np.mean(slice_data)
            dicom_slice.WindowWidth = np.max(slice_data) - np.min(slice_data)
            dicom_slice.PixelSpacing = pixel_spacing
            dicom_slice.SliceThickness = slice_thickness
            dicom_slice.SpacingBetweenSlices = slice_thickness
            dicom_slice.SeriesDescription = f"Slice Thickness: {slice_thickness} mm"
            slice_location = dicom_affine.dot([0, 0, slice_idx, 1])[:3]
            dicom_slice.ImagePositionPatient = [str(x) for x in slice_location]
            dicom_slice.InstanceNumber = slice_idx + 1
            dicom_slice.SOPInstanceUID = f"{folder_name}.{uuid.uuid4()}"

            output_file = os.path.join(output_dir, f"{folder_name}_slice_{slice_idx + 1}.dcm")
            dicom_slice.save_as(output_file)
    except Exception as e:
        logging.error(f"Error converting NIfTI to DICOM for {folder_name}: {e}")

def process_patient_data(patient_path, output_nii_dir, output_dicom_dir, new_spacing):
    try:
        patient_folder = os.path.basename(patient_path)
        dicom_template_path = os.path.join(patient_path, os.listdir(patient_path)[0])
        output_nii_path = os.path.join(output_nii_dir, f"{patient_folder}.nii.gz")
        output_dicom_patient_dir = os.path.join(output_dicom_dir, patient_folder)

        image = read_dicom_series(patient_path)
        if image is None:
            return

        resampled_image = resample_image(image, new_spacing)
        if resampled_image is None:
            return

        save_as_nii(resampled_image, output_nii_path)
        convert_nii_to_dicom(output_nii_path, dicom_template_path, output_dicom_patient_dir, patient_folder)
    except Exception as e:
        logging.error(f"Error processing patient data for {patient_path}: {e}")

def main(input_dir, output_nii_dir, output_dicom_dir, new_spacing):
    if not os.path.exists(output_nii_dir):
        os.makedirs(output_nii_dir)
    if not os.path.exists(output_dicom_dir):
        os.makedirs(output_dicom_dir)

    patient_folders = [os.path.join(input_dir, folder) for folder in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, folder))]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_patient_data, patient_path, output_nii_dir, output_dicom_dir, new_spacing): patient_path for patient_path in patient_folders}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing patients"):
            patient_path = futures[future]
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error in thread for {patient_path}: {e}")

    print(f"Processing complete. NIfTI files saved to: {output_nii_dir}")
    print(f"DICOM series saved to: {output_dicom_dir}")


if __name__ == "__main__":
    input_dir = r'C:\data\CQ500\dicom20240702'
    output_nii_dir = r'C:\data\CQ500\nii_files'
    output_dicom_dir = r'C:\data\CQ500\converted_dicom'
    new_spacing = [1.0, 1.0, 5.0]
    main(input_dir, output_nii_dir, output_dicom_dir, new_spacing)
