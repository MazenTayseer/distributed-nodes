import queue
import json
import os
from WorkerThread import WorkerThread

def worker(comm, images_list):
    CPU_NUM = comm.Get_size() - 1
    rank = comm.Get_rank()
    print(f"node: Worker {rank} starting with {CPU_NUM} CPUs.")

    images_list = sorted(images_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    integer_part_per_cpu = len(images_list) // CPU_NUM
    remainder = len(images_list) % CPU_NUM
    image_counts = [integer_part_per_cpu] * CPU_NUM

    for i in range(remainder):
        image_counts[i] += 1

    num_threads = image_counts[rank - 1]
    print(f"node: Worker {rank} will process {num_threads} images.")

    image_data_path = os.getenv("IMAGE_DATA_PATH", "/home/mazen/gui/image_data.json")
    with open(image_data_path) as f:
        image_data = json.load(f)

    image_data = sorted(image_data, key=lambda x: int(x['file_path'].split('_')[-1].split('.')[0]))

    task_queue = queue.Queue()
    starting_index = sum(image_counts[:rank - 1])

    for i in range(starting_index, starting_index + num_threads):
        task_queue.put((images_list[i], image_data[i]["operation"], i))

    for _ in range(num_threads):
        task_queue.put(None)

    threads = []
    for _ in range(num_threads):
        thread = WorkerThread(task_queue, comm)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    print(f"node: Worker {rank} finished processing tasks.")
