import numpy as np
import cv2

class CrownDetector:
    GRID_ROWS = 5
    GRID_COLS = 5
    
    # Initialize the CrownDetector with template path and CLAHE instance
    def __init__(self, template_path=r"miniprojekt\KD_dataset\pics\Crown_k.png"):
        self.template_path = template_path
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._hsv_templates = None
        self._gray_templates = None
    
    # Enhance image contrast and sharpness using CLAHE and Gaussian blur, handles both grayscale and HSV
    def enhance(self, image):
        if image.ndim == 2:
            eq = self._clahe.apply(image)
        elif image.ndim == 3 and image.shape[2] == 3:
            h, s, v = cv2.split(image)
            v = self._clahe.apply(v)
            eq = cv2.merge((h, s, v))
        else:
            return None
        blurred = cv2.GaussianBlur(eq, (0, 0), 1.0)
        sharp = cv2.addWeighted(eq, 1.5, blurred, -0.5, 0)
        return sharp
    
    # Convert tile to HSV and grayscale and enhance both versions
    def process_tile(self, tile):
        hsv_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
        hsv_tile = self.enhance(hsv_tile)
        gray_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        gray_tile = self.enhance(gray_tile)
        return hsv_tile, gray_tile
    
    # Load and process the template image in every 90 degree orientation, saved in two lists
    def load_and_process_templates(self):
        if self._hsv_templates is not None and self._gray_templates is not None:
            print("Using cached templates")
            return self._hsv_templates, self._gray_templates
            
        template = cv2.imread(self.template_path)
        hsv_template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
        hsv_template = self.enhance(hsv_template)
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        gray_template = self.enhance(gray_template)

        orientations = {
            0: None,
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE
        }
        
        hsv_templates, gray_templates = [], []
        for angle in (0, 90, 180, 270):
            orientation = orientations[angle]
            if orientation is None:
                hsv_templates.append(hsv_template)
                gray_templates.append(gray_template)
            else:
                hsv_templates.append(cv2.rotate(hsv_template, orientation))
                gray_templates.append(cv2.rotate(gray_template, orientation))
                
        # Cache templates
        self._hsv_templates = hsv_templates
        self._gray_templates = gray_templates
        return hsv_templates, gray_templates
    
    # Calculate region of interest coordinates based on angle, with template size and extra distance
    def edge_roi_coords(self, height, width, template_height, template_width, angle, extra_distance=15):
        if angle == 0:
            return 0, extra_distance + template_height, 0, width
        if angle == 180:
            return height - (extra_distance + template_height), height, 0, width
        if angle == 90:
            return 0, height, width - (extra_distance + template_width), width
        return 0, height, 0, extra_distance + template_width
    
    # Find template matches in region of interest above threshold
    def find_matches_in_roi(self, roi, template, threshold):
        matches = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        dilated_matches = cv2.dilate(matches, np.ones((3, 3), np.uint8))
        matches_y, matches_x = np.where((matches == dilated_matches) & (matches >= threshold))
        return list(zip(matches_x, matches_y)), matches[matches_y, matches_x]
    
    # Find crown matches in tiles using templates
    def find_crowns(self, tiles, templates, threshold):
        # If input is a single tile, wrap in a list
        if isinstance(tiles, np.ndarray):
            tiles = [tiles]

        results = []
        for tile_index, tile in enumerate(tiles):
            H, W = tile.shape[:2]
            for angle, template in zip((0, 90, 180, 270), templates):
                template_height, template_width = template.shape[:2]
                y0, y1, x0, x1 = self.edge_roi_coords(H, W, template_height, template_width, angle)
                y0, y1 = max(0, y0), min(H, y1)
                x0, x1 = max(0, x0), min(W, x1)
                if (y1 - y0) < template_height or (x1 - x0) < template_width:
                    continue
                roi = tile[y0:y1, x0:x1]
                
                locations, scores = self.find_matches_in_roi(roi, template, threshold)
                for (match_x, match_y), score in zip(locations, scores):
                    top_left = (x0 + match_x, y0 + match_y)
                    bottom_right = (top_left[0] + template_width, top_left[1] + template_height)
                    results.append({
                        "tile_index": tile_index,
                        "angle": angle,
                        "score": float(score),
                        "rect": (top_left, bottom_right)
                    })
        return results
    
    # Find crowns using HSV and grayscale detection modes
    def find_all_crowns(self, modes, threshold):
        all_results = []
        for mode, (tiles, templates) in modes.items():
            matches = self.find_crowns(tiles, templates, threshold)
            for m in matches:
                m["mode"] = mode
                all_results.append(m)
        return all_results
    
    # Method to remove 'duplicate' matches
    def dedupe_exact(self, matches):
        seen = {}  # rect_key -> match
        for m in matches:
            # key = (x0, y0, x1, y1)
            key = tuple(int(c) for pt in m['rect'] for c in pt)
            if key not in seen or m['score'] > seen[key]['score']:
                seen[key] = m
        return list(seen.values())

    # Main function when processing a single tile
    def main_tile(self, tile, label):
        # Early exit for terrain which never has crowns
        if label in ("home", "table"):
            return 0
        
        # Load and process the template
        hsv_templates, gray_templates = self.load_and_process_templates()

        # Process the tile
        hsv_tile, gray_tile = self.process_tile(tile)

        # Find matches in both HSV and grayscale
        modes = {
            'hsv':  (hsv_tile, hsv_templates),
            'gray': (gray_tile, gray_templates)
        }

        # Find all matches in the tile
        all_matches = self.find_all_crowns(modes, threshold=0.35)

        # Merge and suppress duplicates
        final = self.dedupe_exact(all_matches)

        # Limit the output based on the terrain type
        if label in ('field', 'forest', 'lake'):
            return min(len(final), 1)

        if label in ('grassland', 'swamp'):
            return min(len(final), 2)

        if label == 'mine':
            return min(len(final), 3)
        else:
            print(f"Undefined label: {label}")
            return None
    
### Functions for running on board without labels, used for testing and tuning ###
    # Load and process entire board image, splitting it into tiles
    def load_and_process_board(self, image_path):
        original = cv2.imread(image_path)
        
        # Get actual dimensions from the image
        H, W = original.shape[:2]
        
        hsv_board = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        hsv_board = self.enhance(hsv_board)
        gray_board = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray_board = self.enhance(gray_board)

        tile_height = H // self.GRID_ROWS
        tile_width = W // self.GRID_COLS
        hsv_tiles, gray_tiles = [], []
        for i in range(self.GRID_ROWS):
            y0, y1 = i*tile_height, (i+1)*tile_height
            for j in range(self.GRID_COLS):
                x0, x1 = j*tile_width, (j+1)*tile_width
                hsv_tiles.append(hsv_board[y0:y1, x0:x1])
                gray_tiles.append(gray_board[y0:y1, x0:x1])
        return original, hsv_tiles, gray_tiles

    def draw_results(self, board, results):
        # Draw detection results on the board image
        H, W = board.shape[:2]
        tile_height = H // self.GRID_ROWS
        tile_width = W // self.GRID_COLS
        
        for r in results:
            tidx = r["tile_index"]
            
            # Extract actual row and column based on how tiles were created
            row = tidx // self.GRID_COLS
            col = tidx % self.GRID_COLS
            
            x_off = col * tile_width
            y_off = row * tile_height
            
            (lx0, ly0), (lx1, ly1) = r["rect"]
            tl = (int(lx0 + x_off), int(ly0 + y_off))
            br = (int(lx1 + x_off), int(ly1 + y_off))
            
            cv2.rectangle(board, tl, br, (0,255,0), 2)
            cv2.putText(board, f"{r['score']:.2f}@{r['angle']}°", (tl[0], tl[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
        
        return board

    #Main function when processing the entire board
    def main_board(self, board_image_path):
        # Load and process the board
        print(f"Processing board...")
        original_board, hsv_tiles, gray_tiles = self.load_and_process_board(board_image_path)
        
        # Load templates
        print(f"Loading templates...")
        hsv_templates, gray_templates = self.load_and_process_templates()
        
        # Process all tiles
        all_results = []
        
        for i, (hsv_tile, gray_tile) in enumerate(zip(hsv_tiles, gray_tiles)):
            print(f"Processing tile {i+1}/{len(hsv_tiles)}...")
            # Find crowns in the tile
            modes = {
                'hsv':  ([hsv_tile], hsv_templates),  # Note the list wrapping
                'gray': ([gray_tile], gray_templates)
            }
            matches = self.find_all_crowns(modes, threshold=0.35)
            
            # Make sure all matches have the correct tile index
            for match in matches:
                match['tile_index'] = i  # Explicitly set the tile index
            
            # Process tile results
            hsv_matches = [m for m in matches if m['mode'] == 'hsv']
            gray_matches = [m for m in matches if m['mode'] == 'gray']
            
            final_matches = self.dedupe_exact(hsv_matches + gray_matches)
            all_results.extend(final_matches)
        
        # Draw results on the board
        print(f"Matches found: {len(all_results)}")
        print(f"Drawing results...")
        result_board = self.draw_results(original_board.copy(), all_results)
        
        cv2.imshow("Detected Crowns", result_board)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return result_board