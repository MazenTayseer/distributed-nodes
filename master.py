import cv2  # OpenCV for image processing
from mpi4py import MPI
import os
import paramiko

SERVER_NODE_HOST = "server"
NODE_PORT = 22
NODE_USERNAME = "mazen"
NODE_PASSWORD = "MazenAzure2002"

SERVER_NODE_REMOTE_PATH = "/home/mazen/flask/static/results"

UPLOADS_DIR = "/home/mazen/gui/uploads"
RESULTS_DIR = "/home/mazen/gui/results"


def master(comm, images_list):
    print("Master: Starting to receive results from workers...")
    for i in range(len(images_list)):
        received_data = comm.recv(source=MPI.ANY_SOURCE, tag=i)
        print(f"node:Master: Received data for task {i} from worker.")

        if isinstance(received_data, list):
            serve_preds(received_data, images_list[i])
        else:
            serve_images(received_data, images_list[i])
        print(f"node:Master: Processed result for task {i}.")
    print("node:Master: Finished processing all results.")



def save_image(image, filepath):
    cv2.imwrite(filepath, image)


def serve_preds(received_data, img):
    final_preds = received_data
    for f in final_preds:
        print(f"pred:{img.split('/')[-1]}-{f}%")


def serve_images(recieved_data, img):
    processed_image = recieved_data

    image_name = img.split("/")[-1]

    results_local_path = os.path.join(RESULTS_DIR, image_name)

    save_image(processed_image, results_local_path)

    remote_path = os.path.join(SERVER_NODE_REMOTE_PATH, image_name)
    upload_to_node(
        results_local_path,
        remote_path,
        SERVER_NODE_HOST,
        NODE_PORT,
        NODE_USERNAME,
        NODE_PASSWORD,
    )

    print(f"image:{image_name}")


def upload_to_node(local_path, remote_path, host, port, username, password):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port, username, password)

    # SCPClient takes a paramiko transport as its only argument
    scp = paramiko.SFTPClient.from_transport(ssh.get_transport())
    scp.put(local_path, remote_path)

    scp.close()
    ssh.close()
