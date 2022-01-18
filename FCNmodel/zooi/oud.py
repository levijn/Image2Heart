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


def create_3d_scatterplot(results):
    segmented_images = []
    for i in range(results.size(dim=0)):
        segmented_images.append(create_segmentated_img(results[i,:,:,:]))
    
    fig = plt.figure(1)
    ax = fig.add_subplot(projection="3d")
    colors = ["r", "b", "g"]
    for k in range(len(segmented_images)):
        img = segmented_images[k]
        for i in range(0, img.shape[0], 5):
            for j in range(0, img.shape[1], 5):
                if img[i,j] != 0:
                    ax.scatter(i, j, k*20, c=colors[int(img[i,j])-1], marker="o")
    plt.show()


def create_3d_scatterplot_labels(labels):
    list_labels = []
    for i in range(labels.size(dim=0)):
        list_labels.append(labels[i,:,:].numpy())
    
    fig = plt.figure(1)
    ax = fig.add_subplot(projection="3d")
    colors = ["r", "b", "g"]
    for k in range(len(list_labels)):
        img = list_labels[k]
        for i in range(0, img.shape[0], 5):
            for j in range(0, img.shape[1], 5):
                if img[i,j] != 0:
                    ax.scatter(i, j, k*20, c=colors[int(img[i,j])-1], marker="o")
    plt.show()