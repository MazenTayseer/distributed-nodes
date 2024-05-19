from WorkerThread import WorkerThread
import queue
import json
import time
import os

UPLOADS_DIR = "/home/mazen/gui/uploads"
RESULTS_DIR = "/home/mazen/gui/results"



def worker(comm, images_list):
    CPU_NUM = comm.Get_size() - 1

    threads = []
    integerPartPerPart = len(images_list) // CPU_NUM
    remainder = len(images_list) % CPU_NUM

    result = [integerPartPerPart] * CPU_NUM

    for i in range(remainder):
        result[i] += 1

    number_of_threads = result[comm.Get_rank() - 1]

    with open("/home/mazen/gui/image_data.json") as f:
        image_data = json.load(f)
        
    image_data = sorted(image_data, key=lambda x: x['file_path'])
    images_list = sorted(images_list)

    task_queue = queue.Queue()
    starting_index = (comm.Get_rank() - 1) * number_of_threads
    
    for i in range(starting_index, starting_index + number_of_threads):
        task_queue.put((images_list[i], image_data[i]["operation"], i))


    for i in range(number_of_threads):
        task_queue.put(None)

    for i in range(number_of_threads):
        thread = WorkerThread(task_queue, comm)
        thread.start()
        # print(f"Thread {i+1} started on node", comm.Get_rank())
        threads.append(thread)

    for thread in threads:
        thread.join()

    # time.sleep(4)

    # for i in range(len(images_list)):
    #     try:
    #         image_name = images_list[i].split("/")[-1]

    #         results_local_path = os.path.join(RESULTS_DIR, image_name)
    #         uploads_local_path = os.path.join(UPLOADS_DIR, image_name)

    #         os.remove(results_local_path)
    #         os.remove(uploads_local_path)
    #     except FileNotFoundError:
    #         print("File not found")
            

    # os.remove("/home/mazen/gui/image_data.json")
