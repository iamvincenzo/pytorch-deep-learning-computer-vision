import torch
import cv2 as cv
import numpy as np

if __name__ == "__main__":
    # # load the model
    # model = torch.hub.load(repo_or_dir="ultralytics/yolov5", model="yolov5s", pretrained=True)
    # model.eval()
    # print(model)

    # # inference with the model
    # results = model("https://ultralytics.com/images/zidane.jpg")
    # print(results)
    # results.show()

    # load the model
    model = torch.hub.load(repo_or_dir="ultralytics/yolov5", model="custom",
                           path="./yolov5/runs/train/exp5/weights/best.pt", force_reload=True)
    model.eval()

    cv.namedWindow(winname="YOLO", flags=cv.WINDOW_NORMAL)

    # webcam real time prediction
    cap = cv.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        result = model(frame)

        cv.imshow(winname="YOLO", mat=np.squeeze(result.render()))

        if cv.waitKey(1) == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()
