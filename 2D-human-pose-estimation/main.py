from pathlib import Path
from os.path import join
from scipy.io import loadmat

DATA_PATH = Path("./data/mpii_human_pose_v1_u12_2")

if __name__ == "__main__":
    file_path = DATA_PATH / "mpii_human_pose_v1_u12_2.mat"
    file_path = file_path.resolve()  # Get the absolute path

    annotations = loadmat(file_path)
    print(annotations)
    