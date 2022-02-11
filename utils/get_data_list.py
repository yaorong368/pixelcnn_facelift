import os

def get_data_list(data_list, num_items = None):
    Addiction_mri = '/data/users2/yxiao11/data/Addiction/mri_data/'
    Addiction_mask = '/data/users2/yxiao11/data/Addiction/mask/'

    COBRE_mri = '/data/users2/yxiao11/data/COBRE_slice/mri_data/'
    COBRE_mask = '/data/users2/yxiao11/data/COBRE_slice/mask/'
    
    BSNIP_mri = '/data/users2/yxiao11/data/BSNIP01/mri_data/'
    BSNIP_mask = '/data/users2/yxiao11/data/BSNIP01/mask/'
    
    ABCD_mri = '/data/users2/yxiao11/data/ABCD/mri_data/'
    ABCD_mask = '/data/users2/yxiao11/data/ABCD/mask/'
    
    Add_mri_deface = '/data/users2/yxiao11/data/add_mrideface/mri_data/'
    Add_mri_deface_mask = '/data/users2/yxiao11/data/add_mrideface/mask/'

    
    file_BSNIP = sorted(os.listdir(BSNIP_mri))
    file_COBRE = sorted(os.listdir(COBRE_mri))
    file_ABCD = sorted(os.listdir(ABCD_mri))
    file_Addiction = sorted(os.listdir(Addiction_mri))
    file_add_mri_deface = sorted(os.listdir(Add_mri_deface))

    image_dataset = []
    mask_dataset = []

    for dataset in data_list:
        if dataset == 'Add_mri_deface':
            for i in file_add_mri_deface[0:num_items]:
                sub_path_img = Add_mri_deface + i
                image_dataset.append(
                    os.path.join(sub_path_img)
                )

                sub_path_mask = Add_mri_deface_mask + i
                mask_dataset.append(
                    os.path.join(sub_path_mask)
                )
        
        if dataset == 'Addiction':
            for i in file_Addiction[0:num_items]:
                sub_path_img = Addiction_mri + i
                image_dataset.append(
                    os.path.join(sub_path_img)
                )

                sub_path_mask = Addiction_mask + i
                mask_dataset.append(
                    os.path.join(sub_path_mask)
                )
                
        if dataset == 'ABCD':
            for i in file_ABCD[0:num_items]:
                sub_path_img = ABCD_mri + i
                image_dataset.append(
                    os.path.join(sub_path_img)
                )

                sub_path_mask = ABCD_mask + i
                mask_dataset.append(
                    os.path.join(sub_path_mask)
                )
            
        if dataset == 'BSNIP':
            for i in file_BSNIP[0:num_items]:
                sub_path_img = BSNIP_mri + i
                image_dataset.append(
                    os.path.join(sub_path_img)
                )

                sub_path_mask = BSNIP_mask + i
                mask_dataset.append(
                    os.path.join(sub_path_mask)
                )
                
                
        if dataset == 'COBRE':
            for i in file_COBRE[0:num_items]:
                sub_path_img = COBRE_mri + i
                image_dataset.append(
                    os.path.join(sub_path_img)
                )

                sub_path_mask = COBRE_mask + i
                mask_dataset.append(
                    os.path.join(sub_path_mask)
                )
    return image_dataset, mask_dataset

def get_3d_data_list(data_list=None, num_items = None):
    Addiction_mri = '/data/users2/yxiao11/data/Addiction_3d/mri_data/'
    Addiction_mask = '/data/users2/yxiao11/data/Addiction_3d/mask/'

    # COBRE_mri = '/data/users2/yxiao11/data/COBRE_slice/mri_data/'
    # COBRE_mask = '/data/users2/yxiao11/data/COBRE_slice/mask/'
    
    # BSNIP_mri = '/data/users2/yxiao11/data/BSNIP01/mri_data/'
    # BSNIP_mask = '/data/users2/yxiao11/data/BSNIP01/mask/'
    
    # ABCD_mri = '/data/users2/yxiao11/data/ABCD/mri_data/'
    # ABCD_mask = '/data/users2/yxiao11/data/ABCD/mask/'

    
    # file_BSNIP = sorted(os.listdir(BSNIP_mri))
    # file_COBRE = sorted(os.listdir(COBRE_mri))
    # file_ABCD = sorted(os.listdir(ABCD_mri))
    file_Addiction = sorted(os.listdir(Addiction_mri))

    image_dataset = []
    mask_dataset = []
    
    for i in file_Addiction[0:num_items]:
        sub_path_img = Addiction_mri + i
        image_dataset.append(
            os.path.join(sub_path_img)
        )

        sub_path_mask = Addiction_mask + i
        mask_dataset.append(
            os.path.join(sub_path_mask)
        )
    

    # for dataset in data_list:
    #     if dataset == 'Addiction':
    #         for i in file_Addiction[0:num_items]:
    #             sub_path_img = Addiction_mri + i
    #             image_dataset.append(
    #                 os.path.join(sub_path_img)
    #             )

    #             sub_path_mask = Addiction_mask + i
    #             mask_dataset.append(
    #                 os.path.join(sub_path_mask)
    #             )
                
    #     if dataset == 'ABCD':
    #         for i in file_ABCD[0:num_items]:
    #             sub_path_img = ABCD_mri + i
    #             image_dataset.append(
    #                 os.path.join(sub_path_img)
    #             )

    #             sub_path_mask = ABCD_mask + i
    #             mask_dataset.append(
    #                 os.path.join(sub_path_mask)
    #             )
            
    #     if dataset == 'BSNIP':
    #         for i in file_BSNIP[0:num_items]:
    #             sub_path_img = BSNIP_mri + i
    #             image_dataset.append(
    #                 os.path.join(sub_path_img)
    #             )

    #             sub_path_mask = BSNIP_mask + i
    #             mask_dataset.append(
    #                 os.path.join(sub_path_mask)
    #             )
                
                
    #     if dataset == 'COBRE':
    #         for i in file_COBRE[0:num_items]:
    #             sub_path_img = COBRE_mri + i
    #             image_dataset.append(
    #                 os.path.join(sub_path_img)
    #             )

    #             sub_path_mask = COBRE_mask + i
    #             mask_dataset.append(
    #                 os.path.join(sub_path_mask)
    #             )
    return image_dataset, mask_dataset