import cv2
import numpy as np
import time
import os

pattern_size = (9, 6)

square_size = 3.0
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
objpoints = []
imgpoints = []
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
if not cap.isOpened():
    print("Nie można otworzyć kamery")
    exit()

if not os.path.exists("kalibracja_zrzuty"):
    os.makedirs("kalibracja_zrzuty")

last_capture_time = 0
calibration_done = False
camera_matrix = None
dist_coeffs = None

print("Instrukcja:")
print("- Pokazuj szachownicę 9x6 z różnych perspektyw")
print("- Program automatycznie łapie klatki gdy wykryje szachownicę (max 1 na sekundę)")
print("- Naciśnij 's' aby zapisać parametry kalibracji")
print("- Naciśnij 'q' aby zakończyć program")
print("- Naciśnij 'r' aby zresetować kalibrację i zacząć od nowa")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Nie można pobrać klatki")
        break

    display_frame = frame.copy()
    h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    cv2.putText(display_frame, f"Zebrano klatek: {len(objpoints)}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if calibration_done:
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
        cv2.putText(undistorted, "Wyprostowany", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Wyprostowany obraz", undistorted)

    current_time = time.time()
    ret_chess, corners = cv2.findChessboardCorners(
        gray, pattern_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret_chess:
        cv2.drawChessboardCorners(display_frame, pattern_size, corners, ret_chess)

        if current_time - last_capture_time > 1.0:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            objpoints.append(objp)
            imgpoints.append(corners2)
            capture_filename = f"kalibracja_zrzuty/klatka_{len(objpoints)}.jpg"
            cv2.imwrite(capture_filename, frame)

            print(f"Klatka kalibracyjna {len(objpoints)} zapisana")
            last_capture_time = current_time
            if len(objpoints) >= 5:
                print("Kalibruję kamerę...")
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints, imgpoints, gray.shape[::-1], None, None
                )
                mean_error = 0
                for i in range(len(objpoints)):
                    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                    mean_error += error

                print(f"Parametry kalibracji zaktualizowane, błąd średni: {mean_error / len(objpoints)}")
                print(f"Camera matrix:\n{mtx}")
                print(f"Distortion coefficients: {dist.ravel()}")
                camera_matrix = mtx
                dist_coeffs = dist
                calibration_done = True
                cv2.putText(display_frame, f"Kalibracja zaktualizowana Błąd: {mean_error / len(objpoints):.4f}",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Kalibracja live", display_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and calibration_done:
        np.save("camera_matrix.npy", camera_matrix)
        np.save("dist_coeffs.npy", dist_coeffs)
        print("Zapisano parametry kalibracji")
    elif key == ord('r'):
        objpoints = []
        imgpoints = []
        calibration_done = False
        print("Kalibracja zresetowana")

cap.release()
cv2.destroyAllWindows()