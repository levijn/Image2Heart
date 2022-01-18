import sys
import os
from pathlib import Path

#add the parent folder to the path so modules can be imported
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
sys.path.append(os.path.join(parent_dir, "Helperfunctions"))

import config
from to_image import run_model_rtrn_results, convert_to_segmented_imgs
from patientdataset import get_patient_dataloader
from voxelplot import create_voxelplot_from_results


def main():
    patient = "086"     # Always ues a 3 digit number ('002', '045', etc.)
    img_path = config.simpledata_dir / f"patient{patient}_frame01.nii.gz"
    lbl_path = config.simpledata_dir / f"patient{patient}_frame01_gt.nii.gz"
    
    model_name = "weights_lr10_e15.h5"     # File of the model that should be used
    
    print(f"Generating output for patient {patient}")
    
    patient_dataloader = get_patient_dataloader(img_path, lbl_path)
    
    patient_batch = None
    for batch in patient_dataloader:
        patient_batch = batch
    
    results = run_model_rtrn_results(patient_batch["image"], os.path.join(current_dir, model_name))
    result_images = convert_to_segmented_imgs(results)
    create_voxelplot_from_results(result_images)
    

if __name__ == '__main__':
    main()