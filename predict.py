#-----------------------------------------------------------------------#
#   predict.py integrates single image prediction, video detection, FPS
#   testing, and directory traversal detection into a single script.
#   The mode can be specified to switch between different functionalities.
#-----------------------------------------------------------------------#
import os
import time

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from model import FOCUS  # Replace with your custom model class

def main():
    # Initialize the model
    model = FOCUS()

    #----------------------------------------------------------------------------------------------------------#
    #   mode specifies the test mode:
    #   'predict'            Single image prediction. You can modify the prediction process, such as saving images,
    #                        cropping objects, etc. See detailed comments below.
    #   'video'              Video detection. Can detect from a camera or video file. See detailed comments below.
    #   'fps'                FPS testing. Uses the image in 'img/street.jpg'. See detailed comments below.
    #   'dir_predict'        Traverses a directory for detection and saves the results. Defaults to 'img/' for input
    #                        and 'img_out/' for output. See detailed comments below.
    #   'heatmap'            Visualizes the prediction heatmap. See detailed comments below.
    #   'export_onnx'        Exports the model to ONNX format. Requires PyTorch 1.7.1 or higher.
    #----------------------------------------------------------------------------------------------------------#
    mode = "predict"

    #-------------------------------------------------------------------------#
    #   crop                 Specifies whether to crop objects after single image prediction.
    #   count                Specifies whether to count objects after single image prediction.
    #   crop and count are only valid when mode='predict'.
    #-------------------------------------------------------------------------#
    crop = False
    count = False

    #----------------------------------------------------------------------------------------------------------#
    #   video_path           Specifies the video path. Set video_path=0 to detect from a camera.
    #                        To detect a video file, set video_path="xxx.mp4".
    #   video_save_path      Specifies the path to save the processed video. Set to "" to disable saving.
    #                        To save the video, set video_save_path="yyy.mp4".
    #   video_fps            Specifies the FPS of the saved video.
    #
    #   video_path, video_save_path, and video_fps are only valid when mode='video'.
    #   Saving a video requires exiting with ctrl+c or running until the last frame.
    #----------------------------------------------------------------------------------------------------------#
    video_path = 0
    video_save_path = ""
    video_fps = 25.0

    #----------------------------------------------------------------------------------------------------------#
    #   test_interval        Specifies the number of detections for FPS testing. Higher values improve accuracy.
    #   fps_image_path       Specifies the image used for FPS testing.
    #
    #   test_interval and fps_image_path are only valid when mode='fps'.
    #----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path = "img/street.jpg"

    #-------------------------------------------------------------------------#
    #   dir_origin_path      Specifies the directory path for input images.
    #   dir_save_path        Specifies the directory path for saving processed images.
    #
    #   dir_origin_path and dir_save_path are only valid when mode='dir_predict'.
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path = "img_out/"

    #-------------------------------------------------------------------------#
    #   heatmap_save_path    Specifies the path to save the heatmap. Defaults to 'model_data/heatmap_vision.png'.
    #
    #   heatmap_save_path is only valid when mode='heatmap'.
    #-------------------------------------------------------------------------#
    heatmap_save_path = "model_data/heatmap_vision.png"

    #-------------------------------------------------------------------------#
    #   simplify             Simplifies the ONNX model.
    #   onnx_save_path       Specifies the path to save the ONNX model.
    #-------------------------------------------------------------------------#
    simplify = True
    onnx_save_path = "model_data/models.onnx"

    #-------------------------------------------------------------------------#
    #   Main Functionality
    #-------------------------------------------------------------------------#
    if mode == "predict":
        """
        1. To save the processed image, use r_image.save("img.jpg"). Modify this in the script.
        2. To get the bounding box coordinates, access 'top', 'left', 'bottom', 'right' in the detect_image function.
        3. To crop objects, use the bounding box coordinates to extract regions from the original image.
        4. To add custom text (e.g., object count), check the predicted class and use draw.text to add text.
        """
        while True:
            img = input('Input image filename: ')
            try:
                image = Image.open(img)
            except Exception as e:
                print(f'Open Error! Try again! {e}')
                continue
            else:
                r_image = model.detect_image(image, crop=crop, count=count)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        if not capture.isOpened():
            raise ValueError("Failed to open the camera or video. Check if the camera is installed or the video path is correct.")

        fps = 0.0
        while True:
            t1 = time.time()
            # Read a frame
            ref, frame = capture.read()
            if not ref:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            frame = Image.fromarray(np.uint8(frame))
            # Perform detection
            frame = np.array(model.detect_image(frame))
            # Convert RGB to BGR for OpenCV display
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1.0 / (time.time() - t1))) / 2
            print(f"FPS: {fps:.2f}")
            frame = cv2.putText(frame, f"FPS: {fps:.2f}", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Video", frame)
            key = cv2.waitKey(1) & 0xFF
            if video_save_path != "":
                out.write(frame)

            if key == 27:  # Press 'Esc' to exit
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print(f"Processed video saved to: {video_save_path}")
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = model.get_FPS(img, test_interval)
        print(f"{tact_time:.4f} seconds, {1 / tact_time:.2f} FPS, @batch_size 1")

    elif mode == "dir_predict":
        if not os.path.exists(dir_save_path):
            os.makedirs(dir_save_path)

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = model.detect_image(image)
                r_image.save(os.path.join(dir_save_path, img_name), quality=95, subsampling=0)

    elif mode == "heatmap":
        while True:
            img = input('Input image filename: ')
            try:
                image = Image.open(img)
            except Exception as e:
                print(f'Open Error! Try again! {e}')
                continue
            else:
                model.detect_heatmap(image, heatmap_save_path)

    elif mode == "export_onnx":
        model.convert_to_onnx(simplify, onnx_save_path)

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx', 'dir_predict'.")

if __name__ == "__main__":
    main()