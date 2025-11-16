import cv2
import scipy as sp
import numpy as np
import tkinter as tk

def apply_saturation_filter(frame, sat_threshold):
    """
    Aplikuje filtr, który zachowuje kolory tylko dla pikseli o nasyceniu (S w HSV) powyżej zadanego progu.
    W przeciwnym razie piksele są w skali szarości.
    """
    # Przekształcenie obrazu do przestrzeni HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Podział na kanały HSV
    h, s, v = cv2.split(hsv)

    # Utworzenie maski dla pikseli o nasyceniu większym niż próg
    mask = s > sat_threshold

    # Obraz w skali szarości
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Odzyskanie koloru tam, gdzie maska spełnia warunek
    filtered_frame = frame.copy()
    filtered_frame[~mask] = cv2.merge((grayscale, grayscale, grayscale))[~mask]

    return filtered_frame, mask


def main():
    sat_threshold = 100  # Próg nasycenia dla pierwszego feedu
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error kamera ni działa koliga.")
        return

    print("naciśnij q aby wyłączyć program")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("klata coś zacięła")
            break


        filtered_frame_1, mask = apply_saturation_filter(frame, sat_threshold)

        cv2.imshow("siema", frame)
       # cv2.imshow("siema1", filtered_frame_1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
