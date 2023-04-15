import cv2


def detect_tiles(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    ret, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    (num_labels, labels, stats, centroids) = cv2.connectedComponentsWithStats(255 - thresh, 8, cv2.CV_32S)
    boxes = []
    for i in range(0, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        boxes.append(((x, y), (x + w, y + h)))
    return boxes, centroids, num_labels, labels
