from shutil import copyfile
import os
folder_name = [f for f in os.listdir('/home/dhc/unet/data/images')]
for i in range(210):
    source_file='/home/dhc/unet/data/'+folder_name[i]+'_occ.nii.gz'
    target_file='/home/dhc/unet/data/images/'+folder_name[i]+'/'+folder_name[i]+'_occ.nii.gz'
    copyfile(source_file, target_file)
