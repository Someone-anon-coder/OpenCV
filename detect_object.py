import cv2
import argparse

def object_detection(image_path: str) -> None:
    """Detect objects in the image at image_path

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
    object_count = 0

    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            object_count += 1

    print(f"Objects Detected: {object_count}")
    cv2.imshow("Object Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection in an Image")
    parser.add_argument("--image", required=True, help="Path to the image file")
    args = parser.parse_args()

    object_detection(args.image)
