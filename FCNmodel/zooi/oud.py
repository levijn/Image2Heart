def load_patient_nif_to_tensor(file):
    pt_array = nib.load(file).get_fdata().astype("int")
    sudo_images = []
    new_h, new_w = 264, 288

    for i in range(pt_array.shape[2]):
        slice = pt_array[:,:,i]
        rgb_img = np.stack([slice]*3, axis=0)
        tensor_img = torch.from_numpy(rgb_img)
        tensor_img = F.convert_image_dtype(tensor_img, dtype=torch.float)
        norm_img = F.normalize(tensor_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        sudo_images.append(norm_img)

    stacked_tensor = torch.stack(sudo_images)
    return stacked_tensor