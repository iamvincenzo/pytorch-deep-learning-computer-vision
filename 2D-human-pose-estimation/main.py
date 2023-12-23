import os
from scipy.io import loadmat

DATA_PATH = "./data/mpii_human_pose_v1_u12_2"

if __name__ == "__main__":
    file_path = os.path.join(DATA_PATH, "mpii_human_pose_v1_u12_1.mat")

    annotations = loadmat(file_path, struct_as_record=False)
    
    release = annotations["RELEASE"]

    object1 = release[0, 0]
    
    print(object1._fieldnames)

    annolist = object1.__dict__['annolist']
    print(annolist, type(annolist), annolist.shape)