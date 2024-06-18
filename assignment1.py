import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def histogram(gray_img):
    histogram = cv2.calcHist([gray_img], [0], None, [256], [0,256])
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(histogram)
    plt.xlim([0, 256])
    plt.show()


def contrast_stretching(gray_img):
    min_intensity= np.min(gray_img)
    max_intensity= np.max(gray_img)
    # print("Min intensity: ", min_intensity)
    # print("Max intensity: ", max_intensity)
    contrast_stretched_image = 255*((gray_img-min_intensity)/(max_intensity-min_intensity))
    contrast_stretched_image = contrast_stretched_image.astype(np.uint8)
    cv2.imshow('Strecthed', contrast_stretched_image) 
    return contrast_stretched_image


def binary_image(contrast_stretched_image):
    _, binary_img = cv2.threshold(contrast_stretched_image, 80, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('Binary', binary_img)
    return binary_img


def open_close(binary_img):
    opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))
    cv2.imshow('Opening', opening)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))
    cv2.imshow('Closing', closing)
    return closing

def extract_image(img_path):
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    histogram(gray_img)
    contrast_stretched_image = contrast_stretching(gray_img)
    binary_img = binary_image(contrast_stretched_image)
    final_img = open_close(binary_img)

    cv2.waitKey(0)   
    # Window shown waits for any key pressing event 
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    # Get the directory path of the current script
    script_dir = os.path.dirname(__file__)

    # Construct the relative path to the image file
    image_path = os.path.join(script_dir, 'input_sat_image.jpg')
    extract_image(image_path)