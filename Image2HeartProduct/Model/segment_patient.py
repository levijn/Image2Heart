import sys
from pathlib import Path

#add the parent folder to the path so modules can be imported
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

import config
from to_image import run_model_rtrn_results, convert_to_segmented_imgs
from patientdataset import get_patient_dataloader
from voxelplot import create_voxelplot_from_results
from slicedataset import (RandomZoom,
                          PadImage,
                          SudoRGB,
                          ToTensor,
                          Normalizer)



    


def main():
    patient = "097"
    img_path = config.simpledata_dir / f"patient{patient}_frame01.nii.gz"
    lbl_path = config.simpledata_dir / f"patient{patient}_frame01_gt.nii.gz"
    
    patient_dataloader = get_patient_dataloader(img_path, lbl_path)
    
    
    patient_batch = None
    for batch in patient_dataloader:
        patient_batch = batch
    
    results = run_model_rtrn_results(patient_batch["image"])
    result_images = convert_to_segmented_imgs(results)
    create_voxelplot_from_results(result_images)
    
    # create_3d_scatterplot(results)
    # create_3d_scatterplot_labels(labels)

    
    
    # fig = plt.figure(1)
    # ax1 = fig.add_subplot(1,5,1)
    # ax1.imshow(F.to_pil_image(slice_result[0,:,:]))
    # ax2 = fig.add_subplot(1,5,2)
    # ax2.imshow(F.to_pil_image(slice_result[1,:,:]))
    # ax3 = fig.add_subplot(1,5,3)
    # ax3.imshow(F.to_pil_image(slice_result[2,:,:]))
    # ax4 = fig.add_subplot(1,5,4)
    # ax4.imshow(F.to_pil_image(slice_result[3,:,:]))
    # ax5 = fig.add_subplot(1,5,5)
    # ax5.imshow(F.to_pil_image(slice_input), cmap="gray")
    # plt.show()


if __name__ == '__main__':
    main()