import cv2
import numpy as np

from puzzle_solver.edge import EdgeType
from puzzle_solver.utils import get_wrapped_contour_section


def find_lock_points(contours, threshold=3):
    """
    Find lock points.

    :param contours: The contour.
    :param threshold: The ratio between the smallest and the largest convexity defect.
    :return: A (lock_points, defects_start_end) tuple where lock_points is an array of
    the same length as contours,containing the index of lock points in contours. defects_start_end
    is an array of the same length as contours, containing the index of the beginning and the end
    of the convexity defect for each lock point.
    """
    lock_points = []
    defect_start_end = []
    for i in range(len(contours)):
        cnt = contours[i]
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)
        defects_size = defects[:, 0, 3] / 256.0
        significant_defects = defects_size > defects_size.max() / threshold

        local_lock_points = []
        local_defect_start_end = []
        for s, e, f, d in defects[significant_defects][:, 0]:
            local_defect_start_end.append([s, e])
            local_lock_points.append(f)
        lock_points.append(local_lock_points)
        defect_start_end.append(local_defect_start_end)
    return lock_points, defect_start_end


def classify_lock_points(contours, lock_points, defects_start_end):
    """
    Classify each lock point as a head, hole or flat edge.

    :param contours: The contours of the puzzle piece.
    :param lock_points: Array of the same length as contours,
    containing the index of lock points in contours.
    :param defects_start_end: Array of the same length as contours,
    containing the index of the beginning and the end of the convexity defect
    for each lock point.

    :return: a tuple containing (head_sections, hole_sections, defect_labels),
    where head_sections contains the sections of the contour surrounding a head,
    hole_sections the ones surrounding a hole and defect_labels an EdgeType array
    giving the type of edge associated to each lock point.
    """
    head_sections = []
    hole_sections = []
    defect_labels = []

    for i in range(len(contours)):
        local_lock_points = lock_points[i]
        puzzle_piece_area = cv2.contourArea(contours[i])
        local_head_section = []
        local_hole_section = []
        local_head_mask = np.full(len(local_lock_points), fill_value=False)
        local_hole_section = []
        defect_labels.append([EdgeType.FLAT] * len(local_lock_points))

        for j in range(len(local_lock_points)):
            defect_start = local_lock_points[j]
            defect_end = local_lock_points[(j + 1) % len(local_lock_points)]
            contour_section = get_wrapped_contour_section(contours[i], defect_start, defect_end)

            # Measure circularity to detect head
            area = cv2.contourArea(contour_section)
            perimeter = cv2.arcLength(contour_section, closed=True)
            circularity = area / perimeter**2

            if abs(circularity - 0.25 / np.pi) < 0.03 and area < 0.15 * puzzle_piece_area:
                local_head_section.append(contour_section)
                local_head_mask[j] = True
                local_head_mask[(j + 1) % len(local_lock_points)] = True
                defect_labels[-1][j] = EdgeType.HEAD
                defect_labels[-1][(j + 1) % len(local_lock_points)] = EdgeType.HEAD
                head_sections.append(local_head_section)

        # If not a head, then a hole
        for j, (is_head, (defect_start, defect_end)) in enumerate(zip(local_head_mask, defects_start_end[i])):
            if not is_head:
                contour_section = get_wrapped_contour_section(contours[i], defect_start, defect_end)
                local_hole_section.append(contour_section)
                area = cv2.contourArea(contour_section)
                defect_labels[-1][j] = EdgeType.HOLE
        hole_sections.append(local_hole_section)

    return head_sections, hole_sections, defect_labels


def detect_corners(img, contours, head_sections, hole_sections):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)

    for local_head_section in head_sections:
        cv2.drawContours(mask, local_head_section, -1, color=0, thickness=cv2.FILLED)
    for local_hole_section in hole_sections:
        cv2.drawContours(mask, local_hole_section, -1, color=255, thickness=cv2.FILLED)
    rect_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corners = []
    for rect_contour in rect_contours:
        corners.append([])
        poly = cv2.approxPolyDP(rect_contour, epsilon=0.1 * 0.25 * cv2.arcLength(rect_contour, True), closed=True)
        for pt in poly[:, 0, :]:
            corners[-1].append(pt)
    corners = np.array(corners)
    return corners
