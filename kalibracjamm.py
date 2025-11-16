import cv2
import numpy as np
import sys
import time

CHECKERBOARD_SIZE = (9, 6)
SQUARE_SIZE_MM = 25.583
CAL_MATRIX_FILE = 'camera_matrix.npy'
DIST_COEFFS_FILE = 'dist_coeffs.npy'

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080



def main():
    print("Ładowanie plików kalibracyjnych...")
    try:
        mtx = np.load(CAL_MATRIX_FILE)
        dist = np.load(DIST_COEFFS_FILE)
        print("Pliki załadowane.")
    except FileNotFoundError as e:
        print(f"\nBŁĄD: Nie ma plików kalibracyjnych: {e.filename}")
        sys.exit(1)

    print("Uruchamianie kamery...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("BŁĄD: Nie można otworzyć kamery.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)


    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Kamera działa w rozdzielczości: {int(width)}x{int(height)}")
    h, w = int(height), int(width)
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    x_roi, y_roi, w_roi, h_roi = roi

    print("\nGotowy do pomiaru.")
    print("Naciśnij 's', aby pobrać klatkę i kalibrować.")
    print("Naciśnij 'q', aby wyjść.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Błąd odczytu klatki.")
            break

        undistorted_frame = cv2.undistort(frame, mtx, dist, None, new_camera_mtx)
        display_frame = undistorted_frame[y_roi: y_roi + h_roi, x_roi: x_roi + w_roi]
        preview = cv2.resize(display_frame, (int(w / 2), int(h / 2)))
        gray_preview = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(gray_preview, CHECKERBOARD_SIZE, None)
        if ret_corners:
            cv2.drawChessboardCorners(preview, CHECKERBOARD_SIZE, corners, ret_corners)

        cv2.putText(preview, "Nacisnij 's' aby obliczyc, 'q' aby wyjsc", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Podgląd korekcji", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("\nKlatka pobrana. Szukam rogów")
            gray_full = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
            ret_corners, corners = cv2.findChessboardCorners(gray_full, CHECKERBOARD_SIZE, None)
            if ret_corners:
                print("Rogi znalezione, kalibruje")
                corners_subpix = cv2.cornerSubPix(gray_full, corners, (11, 11), (-1, -1),
                                                  (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                p1 = corners_subpix[0][0]
                p2 = corners_subpix[1][0]
                pixel_distance = np.linalg.norm(p1 - p2)
                px_per_mm_ratio = pixel_distance / SQUARE_SIZE_MM

                print("\n--- WYNIK ---")
                print(f"  Odległość w pikselach: {pixel_distance:.4f} px")
                print(f"  Rzeczywista odległość: {SQUARE_SIZE_MM} mm")
                print(f"  Stosunek: {px_per_mm_ratio:.4f} px/mm")
                print("---------------\n")
                break
            else:
                print("Nie znaleziono rogów na tej klatce.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()