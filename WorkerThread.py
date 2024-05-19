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
            if task is None:
                break
            image, operation, tag = task
            result = self.process_image(image, operation)

            self.send_result(result, tag)

    def classify_image(self, img_path):
        from keras.preprocessing import image
        from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

        # Load pre-trained VGG16 model
        model = VGG16(weights="imagenet")

        img = image.load_img(
            img_path, color_mode="rgb", target_size=(224, 224)
        )  # VGG16 input size
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Predict class probabilities
        preds = model.predict(x)

        # Decode predictions
        decoded_preds = decode_predictions(preds, top=3)[0]  # top 3 predictions
        final_preds = []
        for i, (imagenet_id, label, prob) in enumerate(decoded_preds):
            final_preds.append(f"{label} ({prob:.2f})")

        return final_preds

    def process_image(self, image, operation):
        # Load the image
        print(
            "node:Machine", self.rank, "Processing", image.split("/")[-1]
        )
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        # Perform the specified operation
        print(f"preimg:{image.split('/')[-1]}")

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

        return result

    def send_result(self, result, tag):
        # Send the result to the master node
        self.comm.send(result, dest=0, tag=tag)
