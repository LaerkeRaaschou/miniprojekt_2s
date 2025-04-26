import cv2
import numpy as np
import pandas as pd

class ImageProcessor:
    def __init__(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path
        self.label_data = pd.read_csv(label_path)

    def load_image(self, index):
        image = cv2.imread(f'{self.image_path}{index}.jpg')
        return image

    def cut_tiles(self, image):
        tiles = []
        for i in range(5):
            tiles.append([])
            for j in range(5):
                tile = image[i*100:(i+1)*100, j*100:(j+1)*100]
                tiles[-1].append(tile)
        return tiles

    def cut_center(self, channel):
        return np.array([
            pixel for i, row in enumerate(channel) for j, pixel in enumerate(row)
            if j <= 25 or j >= 75 or i <= 25 or i >= 75
        ], dtype=np.uint8)

    def hist_feature(self, edge, bins, max_val):
        hist = cv2.calcHist([edge], [0], None, [bins], [0, max_val])
        return cv2.normalize(hist, hist).flatten()

    def hsv_feature(self, tile, bins):
        hsv_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_tile)
        return (
            self.hist_feature(self.cut_center(h), bins, 180),
            self.hist_feature(self.cut_center(s), bins, 255),
            self.hist_feature(self.cut_center(v), bins, 180)
        )

    def sobel_image(self, image):
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        return cv2.magnitude(sobel_x, sobel_y).flatten()

    def texture_feature(self, tile, bins):
        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        cropped = gray[5:-5, 5:-5]
        edges = self.sobel_image(cropped)
        edges = np.clip(edges, 0, 255).astype(np.uint8)
        return self.hist_feature(edges.reshape(-1, 1), bins, 255)

    def process_image(self, image_nr):
        image = self.load_image(image_nr)
        tiles = self.cut_tiles(image)
        labels = self.label_data[self.label_data["Image"] == image_nr]
        features = []
        for j, row in enumerate(tiles):
            row_labels = labels[labels["row"] == j]
            for k, tile in enumerate(row):
                col_labels = row_labels[row_labels["column"] == k]
                if not col_labels.empty:
                    hist_h, hist_s, hist_v = self.hsv_feature(tile, 60)
                    texture = self.texture_feature(tile, 81)
                    true_label = col_labels["TrueLabel"].values[0]
                    crowns = col_labels["Crowns"].values[0]
                    features.append([image_nr, k, j, *hist_h, *hist_s, *hist_v, *texture, true_label, crowns])
        return features

    def create_dataset(self, image_indices, save_path = None):
        dataset = []
        for image_nr in image_indices:
            dataset.extend(self.process_image(image_nr))
        columns = ([ "Image", "Column", "Row"] + 
                  [f"Hist_H_{i}" for i in range(60)] + 
                  [f"Hist_S_{i}" for i in range(60)] + 
                  [f"Hist_V_{i}" for i in range(60)] + 
                  [f"Texture_{i}" for i in range(81)] + 
                  ["TrueLabel", "Crowns"])
        df = pd.DataFrame(dataset, columns=columns)
        if save_path:
            df.to_csv(save_path, index=False)
        return df

def main():
    image_path = "/Users/laerkeraaschou/Desktop/semester2/duas/miniprojekt/KD_dataset/cropped_pics/"
    label_path = "/Users/laerkeraaschou/Desktop/semester2/duas/miniprojekt/labels_med_kroner.csv"
    processor = ImageProcessor(image_path, label_path)
    image_indices = [1, 2, 3]  # Example list of images to process
    dataset = processor.create_dataset(image_indices)
    print(dataset.head())

if __name__ == "__main__":
    main()
