import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from queue import Queue

from preprocessing import *
from visualizer import *
from crown_class import CrownDetector

class PointCounter:
    def __init__(self, grid_shape=(5, 5)):
        self.grid_shape = grid_shape

    def grid_neighbors(self, coord):
        row, col = coord
        potential_neighbors = [
            (row - 1, col), (row + 1, col),
            (row, col - 1), (row, col + 1)
        ]
        return [(r, c) for r, c in potential_neighbors if 0 <= r < self.grid_shape[0] and 0 <= c < self.grid_shape[1]]

    def check_neighbors(self, grid, start):
        frontier = Queue()
        frontier.put(start)
        came_from = {start: None}

        while not frontier.empty():
            current = frontier.get()
            for neighbor in self.grid_neighbors(current):
                if neighbor not in came_from:
                    frontier.put(neighbor)
                    came_from[neighbor] = current

        return came_from

    def bfs(self, start, visited, combined):
        queue = Queue()
        queue.put(start)
        visited.add(start)

        component = []
        start_label = combined[start[0], start[1], 0]

        while not queue.empty():
            r, c = queue.get()
            component.append((r, c))

            for neighbor in self.grid_neighbors((r, c)):
                if neighbor not in visited and combined[neighbor[0], neighbor[1], 0] == start_label:
                    visited.add(neighbor)
                    queue.put(neighbor)

        return component

    def calculate_point(self, labels, crowns):
        total_points = 0
        visited = set()

        combined = np.array(list(zip(labels, crowns)), dtype=object).reshape(self.grid_shape[0], self.grid_shape[1], 2)

        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                if (i, j) in visited:
                    continue

                component = self.bfs((i, j), visited, combined)
                region_size = len(component)
                crowns_sum = sum(combined[r, c, 1] for (r, c) in component)
                total_points += region_size * crowns_sum

        return total_points

def split_data(dataset, train=0.8, test=0.2, seed=39):
    if train + test != 1:
        raise ValueError("The distribution between train and test must be equal to 1")

    np.random.seed(seed)
    if isinstance(dataset, pd.DataFrame):
        if 'Image' not in dataset.columns:
            raise KeyError("'Image' column not found in dataset")
        image_ids = dataset["Image"].unique()
    else:
        raise TypeError("Expected dataset as a pandas DataFrame")
    np.random.shuffle(image_ids)

    total = len(image_ids)
    train_end = int(total * train)

    train_ids = image_ids[:train_end]
    test_ids = image_ids[train_end:]

    train_data = dataset[dataset["Image"].isin(train_ids)]
    test_data = dataset[dataset["Image"].isin(test_ids)]

    return train_data, test_data

def split_feature_target(data):
    tiles = data[["Image", "Column", "Row"]].to_numpy()
    X = data.drop(columns=["Image", "Column", "Row", "TrueLabel", "Crowns"]).to_numpy()
    Y = data[["TrueLabel", "Crowns"]].to_numpy()
    return tiles, X, Y

def load_or_create_dataset(image_path, label_path, data_path, img_range):
    processer = ImageProcessor(image_path, label_path)
    try:
        full_data = pd.read_csv(data_path)
    except FileNotFoundError:
        full_data = processer.create_dataset(list(img_range), data_path)
    return full_data, processer

def prepare_data(full_data):
    train, test = split_data(full_data, 0.8, 0.2, 39)
    tiles_train, X_train, Y_train = split_feature_target(train)
    tiles_test, X_test, Y_test = split_feature_target(test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    encoder = LabelEncoder()
    encoder.fit(full_data["TrueLabel"].values)

    Y_train_encoded = encoder.transform(Y_train[:, 0])
    Y_test_labels_encoded = encoder.transform(Y_test[:, 0])

    return train, test, tiles_train, tiles_test, X_train_scaled, X_test_scaled, Y_train_encoded, Y_test_labels_encoded, encoder, scaler

def train_models(X_train_scaled, Y_train_encoded):
    lda = LinearDiscriminantAnalysis(n_components=len(np.unique(Y_train_encoded)) - 1)
    X_train_lda = lda.fit_transform(X_train_scaled, Y_train_encoded)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_lda, Y_train_encoded)
    return lda, knn

def predict_on_test_set(processer, test, encoder, combined_test, knn, crown_detector):
    image_ids = np.unique(combined_test[:, 0])
    predictions = []

    for img_id in image_ids:
        tiles = combined_test[combined_test[:, 0] == img_id]
        image = processer.load_image(int(img_id))
        img_tiles = processer.cut_tiles(image)
        img_tiles = [elem for row in img_tiles for elem in row]

        img_test_data = test[test["Image"] == img_id].reset_index(drop=True)

        labels = []
        crowns = []
        true_labels = img_test_data["TrueLabel"].values
        true_crowns = img_test_data["Crowns"].values

        for i, tile in enumerate(tiles):
            label_encoded = knn.predict(tile[3:].reshape(1, -1))[0]
            label_decoded = encoder.inverse_transform([label_encoded])[0]
            crown = crown_detector.main_tile(img_tiles[i], label_decoded)

            labels.append(label_decoded)
            crowns.append(crown)

            predictions.append({
                "Image": img_id,
                "Tile_Index": i,
                "Predicted_Label": label_decoded,
                "Predicted_Crown": crown,
                "True_Label": true_labels[i],
                "True_Crown": true_crowns[i]
            })
        
        counter = PointCounter()
        predicted_points = counter.calculate_point(labels, crowns)
        true_points = counter.calculate_point(true_labels, true_crowns)

        predictions.append({
            "Image": img_id,
            "Tile_Index": "Total",
            "Predicted_Label": None,
            "Predicted_Crown": None,
            "True_Label": None,
            "True_Crown": None,
            "Predicted_Points": predicted_points,
            "True_Points": true_points
        })

    predictions_df = pd.DataFrame(predictions)
    return predictions_df

def evaluate_predictions(predictions_df, encoder):
    tile_predictions = predictions_df[predictions_df["Tile_Index"] != "Total"]
    board_predictions = predictions_df[predictions_df["Tile_Index"] == "Total"]

    cm = confusion_matrix(tile_predictions["True_Label"], tile_predictions["Predicted_Label"], labels=encoder.classes_)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=True)
    ax.set_title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()

    rec_labels = recall_score(tile_predictions["True_Label"], tile_predictions["Predicted_Label"], average=None, labels=encoder.classes_)
    per_labels = precision_score(tile_predictions["True_Label"], tile_predictions["Predicted_Label"], average=None, labels=encoder.classes_)

    print(encoder.classes_)
    print(f'recall \n{rec_labels}')
    print(f'precision \n{per_labels}')

    acc_labels = accuracy_score(tile_predictions["True_Label"], tile_predictions["Predicted_Label"])
    acc_crowns = accuracy_score(tile_predictions["True_Crown"], tile_predictions["Predicted_Crown"])
    point_mae = mean_absolute_error(board_predictions["True_Points"], board_predictions["Predicted_Points"])
    point_r2 = r2_score(board_predictions["Predicted_Points"],board_predictions["True_Points"])
    
    print(f'Accuracy for labels: {acc_labels}')
    print(f'Accuracy for crowns: {acc_crowns}')
    print(f'Mean absolute error for points: {point_mae}')
    print(f'R2 score for points: {point_r2}')

def main():
    image_path = "/Users/laerkeraaschou/Desktop/miniprojekt_2s/images/"
    label_path = "/Users/laerkeraaschou/Desktop/miniprojekt_2s/labels.csv"
    data_path = "/Users/laerkeraaschou/Desktop/miniprojekt_2s/full_data.csv"
    template_path = "/Users/laerkeraaschou/Desktop/miniprojekt_2s/images/Crown_k.png"
    img_range = range(1, 75)
    c_d = CrownDetector(template_path)

    full_data, processer = load_or_create_dataset(image_path, label_path, data_path, img_range)
    train, test, tiles_train, tiles_test, X_train_scaled, X_test_scaled, Y_train_encoded, Y_test_labels_encoded, encoder, scaler = prepare_data(full_data)
    lda, knn = train_models(X_train_scaled, Y_train_encoded)

    combined_test = np.column_stack((tiles_test, lda.transform(X_test_scaled)))
    predictions_df = predict_on_test_set(processer, test, encoder, combined_test, knn, c_d)
    
    evaluate_predictions(predictions_df, encoder)

if __name__ == "__main__":
    main()