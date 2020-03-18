
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy
import matplotlib.image as mpimg
import cv2
from PIL import Image
from PIL import ImageDraw
import




def download_and_resize_image(image,
                              new_width=256, new_height=256,
                              display=False):
    # resize image
    pil_image = Image.open(image)
    pil_image = pil_image.resize((new_width, new_height))

    return image


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               thickness=2):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
               (left, top)],
              width=thickness,
              fill=color)


def draw_boxes(image, boxes, class_names, scores, max_boxes=4, min_score=0.05):
    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])

            color = "red"
            image_pil = Image.fromarray(numpy.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
                image_pil,
                ymin,
                xmin,
                ymax,
                xmax,
                color)

            numpy.copyto(image, numpy.array(image_pil))
    return image


def load_img(path):
    img = tf.io.read_file(path)

    img = tf.image.decode_jpeg(img, channels=3)

    return img


def run_detector(detect, path):
    img = load_img(path)

    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

    result = detect(converted_img)

    result = {key: value.numpy() for key, value in result.items()}
    object_info = result['detection_class_entities']
    weapon_indices = []

    for n in range(len(object_info)):

        if (str(object_info[n])[2:-1]) in ["Handgun", "Rifle", "Knife", "Sword"]:
            weapon_indices.append(n)

    result_keys = list(result.keys())

    for key in result_keys:
        result[key] = result[key][weapon_indices]

    image_with_boxes = draw_boxes(
        img.numpy(), result["detection_boxes"],
        result["detection_class_entities"],
        result["detection_scores"])

    return  image_with_boxes
# dump your image here




drone = Tello()  # declaring drone object
drone.connect()
time.sleep(0.5)
drone.takeoff()
drone.move_up(30)
drone.streamon()

while True:
    frame = drone.get_frame_read().frame  # capturing frame from drone
    downloaded_image_path = download_and_resize_image(frame, 1280, 856, True)
    img = run_detector(detector, downloaded_image_path)
    screen = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('frame', screen)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

drone.streamoff()
drone.land()
drone.end()


