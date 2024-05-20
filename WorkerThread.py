import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import threading
import cv2
import numpy as np

class WorkerThread(threading.Thread):
    def __init__(self, task_queue, comm):
        super().__init__()
        self.task_queue = task_queue
        self.rank = comm.Get_rank()
        self.comm = comm

    def run(self):
        while True:
            task = self.task_queue.get()
            print(f"node: Worker {self.rank} received task: {task}")  # Add this line to print received task
            if task is None:
                print(f"node: Machine {self.rank} received termination signal.")
                break
            image, operation, tag = task
            print(f"node: Machine {self.rank} processing task {tag} with operation {operation}.")
            try:
                result = self.process_image(image, operation)
                print(f"node: Machine {self.rank} finished processing task {tag}.")
                self.send_result(result, tag)
            except Exception as e:
                print(f"node: Machine {self.rank} encountered an error processing task {tag}: {e}")

    def classify_image(self, img_path):
        from keras.preprocessing import image
        from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

        print(f"node: Machine {self.rank} loading model for classification.")
        model = VGG16(weights="imagenet")

        img = image.load_img(img_path, color_mode="rgb", target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        print(f"node: Machine {self.rank} predicting image class.")

        preds = model.predict(x)
        decoded_preds = decode_predictions(preds, top=3)[0]
        final_preds = [f"{label} ({prob:.2f})" for (imagenet_id, label, prob) in decoded_preds]

        return final_preds

    def process_image(self, image, operation):
        print(f"node: Machine {self.rank} processing image {os.path.basename(image)} with operation {operation}.")
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        print(f"preimg:{os.path.basename(image)}")

        try:
            if operation == "edge_detection":
                result = cv2.Canny(img, 100, 200)
            elif operation == "color_inversion":
                result = cv2.bitwise_not(img)
            elif operation == "grayscale":
                result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif operation == "blur":
                result = cv2.GaussianBlur(img, (5, 5), 0)
            elif operation == "thresholding":
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, result = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
            elif operation == "dilation":
                kernel = np.ones((5, 5), np.uint8)
                result = cv2.dilate(img, kernel, iterations=1)
            elif operation == "erosion":
                kernel = np.ones((5, 5), np.uint8)
                result = cv2.erode(img, kernel, iterations=1)
            elif operation == "opening":
                kernel = np.ones((5, 5), np.uint8)
                result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            elif operation == "closing":
                kernel = np.ones((5, 5), np.uint8)
                result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            elif operation == "classification":
                result = self.classify_image(image)
            elif operation == "enhancement":
                from ESRGAN import esr
                result = esr.enhance_image(image)
            else:
                raise ValueError(f"Unsupported operation: {operation}")

            return result
        except Exception as e:
            print(f"Error processing image {os.path.basename(image)} with operation {operation}: {e}")
            raise

		
    def send_result(self, result, tag):
        try:
            print(f"node: Machine {self.rank} sending result for task {tag}.")
            self.comm.send(result, dest=0, tag=tag)
        except Exception as e:
            print(f"Error sending result for task {tag}: {e}")
