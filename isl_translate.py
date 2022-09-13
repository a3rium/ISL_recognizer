#!/usr/bin/python

import numpy as np
import argparse
import tensorflow.keras as keras
import matplotlib.image as mpimg
from skimage.transform import resize
import matplotlib.pyplot as plt
import utils


def main():
    parser = argparse.ArgumentParser(
        description="This script applies a pre-trained CNN model to find the word shown in the image input and outputs the title on the image."
    )
    parser.add_argument("i", type=str, help="The input filename.")
    parser.add_argument("o", type=str, help="The output file name.")

    args = parser.parse_args()

    input_path = args.i
    output_path = args.o

    image = mpimg.imread(input_path)

    prediction = make_prediction(image)

    save_image(output_path, prediction, image)

    return


def make_prediction(image):
    processed_image = utils.preprocess_image(image)
    processed_image = resize(processed_image, (256, 256))
    processed_image = processed_image.reshape(
        (1, processed_image.shape[0], processed_image.shape[1], 1)
    )

    print(processed_image.shape)

    Model = keras.models.load_model("predictor_model")

    prediction = Model.predict(processed_image)
    predicted_classes = np.argmax(prediction, axis=-1)
    print(predicted_classes)
    return predicted_classes


def save_image(output_path, prediction, image):
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(str(prediction))
    plt.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
