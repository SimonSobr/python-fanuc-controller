import cv2

for idx in range(6):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    ok, frame = cap.read()
    if ok:
        cv2.putText(frame, f"CAM_INDEX = {idx}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
        cv2.imshow("Camera test (press any key)", frame)
        print("Works:", idx)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cap.release()