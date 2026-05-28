import cv2
import numpy as np
import dlib

class FacialSymmetryAnalyzer:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(
            "shape_predictor_68_face_landmarks.dat"
        )

    def get_face_landmarks(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_faces = self.face_detector(gray_image)

        if len(detected_faces) == 0:
            return None

        landmark_points = []

        for detected_face in detected_faces:
            shape = self.landmark_predictor(gray_image, detected_face)

            for point in range(68):
                x = shape.part(point).x
                y = shape.part(point).y
                landmark_points.append((x, y))

        return landmark_points

    def compute_average_distance(self, landmarks, index_points):
        if landmarks is None:
            return 0

        selected_points = np.array([landmarks[i] for i in index_points])

        average_distance = np.mean(
            np.linalg.norm(
                selected_points - np.flip(selected_points, axis=0),
                axis=1
            )
        )

        return average_distance

    def analyze_symmetry(self, original_landmarks, flipped_landmarks):

        left_face = list(range(0, 17))
        right_face = list(range(16, 0, -1))

        forehead_region = list(range(17, 27))
        jaw_region = left_face + right_face

        original_distance = self.compute_average_distance(
            original_landmarks,
            jaw_region
        )

        flipped_distance = self.compute_average_distance(
            flipped_landmarks,
            jaw_region
        )

        forehead_original = self.compute_average_distance(
            original_landmarks,
            forehead_region
        )

        forehead_flipped = self.compute_average_distance(
            flipped_landmarks,
            forehead_region
        )

        jaw_ratio = original_distance / flipped_distance
        forehead_ratio = forehead_original / forehead_flipped

        return jaw_ratio, forehead_ratio

    def draw_landmark_lines(self, image, landmarks):

        if landmarks is None:
            return

        face_outline = list(range(0, 17))

        for i in range(len(face_outline) - 1):
            cv2.line(
                image,
                landmarks[face_outline[i]],
                landmarks[face_outline[i + 1]],
                (0, 255, 0),
                2
            )

    def run_analysis(self):

        image_path = input("Enter image path: ")

        image = cv2.imread(image_path)

        if image is None:
            print("Unable to load image.")
            return None, None, None

        mirrored_image = cv2.flip(image, 1)

        original_landmarks = self.get_face_landmarks(image)
        mirrored_landmarks = self.get_face_landmarks(mirrored_image)

        if original_landmarks is None or mirrored_landmarks is None:
            print("Face not detected.")
            return None, None, None

        symmetry_values = self.analyze_symmetry(
            original_landmarks,
            mirrored_landmarks
        )

        return image, mirrored_image, symmetry_values


if __name__ == "__main__":

    analyzer = FacialSymmetryAnalyzer()

    original_img, flipped_img, symmetry_result = analyzer.run_analysis()

    if original_img is not None:

        print("\nFacial Symmetry Analysis")
        print("-" * 30)

        jaw_percentage = (1 - symmetry_result[0]) * 100
        forehead_percentage = (1 - symmetry_result[1]) * 100

        print(f"Jaw Symmetry Score: {jaw_percentage:.2f}%")
        print(f"Forehead Symmetry Score: {forehead_percentage:.2f}%")

        original_points = analyzer.get_face_landmarks(original_img)
        analyzer.draw_landmark_lines(original_img, original_points)

        resized_width = int(original_img.shape[1] * 0.5)
        resized_height = int(original_img.shape[0] * 0.5)

        output_image = cv2.resize(
            original_img,
            (resized_width, resized_height)
        )

        cv2.imshow("Facial Symmetry Result", output_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()