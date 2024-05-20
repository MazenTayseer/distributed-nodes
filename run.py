from mpi4py import MPI  # MPI for distributed computing
import os
import json

from master import master
from worker import worker

comm = MPI.COMM_WORLD

TEST_CASES_DIR = "/home/mazen/gui/uploads"
RESULTS_DIR = "/home/mazen/gui/results"


def main():
    for directory in [TEST_CASES_DIR, RESULTS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    images_list = [
        os.path.join(TEST_CASES_DIR, filename)
        for filename in os.listdir(TEST_CASES_DIR)
        if any(
            filename.lower().endswith(ext)
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]
        )
    ]

    with open("/home/mazen/gui/image_data.json") as f:
        image_data = json.load(f)

    def matches_filename(data, image_path):
        if data.split("/")[-1] == image_path.split("/")[-1]:
            return True
        return False

    final_image_list = []
    for img in images_list:
        for d in image_data:
            if matches_filename(d["file_path"], img):
                final_image_list.append(img)

    if comm.Get_rank() == 0:
        master(comm, final_image_list)
    else:
        worker(comm, final_image_list)


main()
