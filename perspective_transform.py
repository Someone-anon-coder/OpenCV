import cv2
import numpy as np
import argparse

def perspective_transform(image_path: str) -> None:
    """Descewe the image at image_path

    Args:
        image_path (str): Path to the image file
    """
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    window_width = 480
    window_height = 320
    image = cv2.resize(image, (window_width, window_height))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)

            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            width = int(rect[1][0] - rect[0][0])
            height = int(rect[3][1] - rect[0][1])

            dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (width, height))

            cv2.imshow("Warped Document", warped)
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)

    cv2.imshow("Perspective Transform", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perspective Transform of an Image")
    parser.add_argument("--image", required=True, help="Path to the image file")
    args = parser.parse_args()

    perspective_transform(args.image)