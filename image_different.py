import sys
import cv2

# --- Recommended Parameters ---
# The maximum number of features to detect in each image.
N_FEATURES = 1500
# The threshold for filtering good matches using Lowe's ratio test.
GOOD_MATCH_THRESHOLD = 0.75
# The similarity score threshold (in percent) to consider images as similar.
SIMILARITY_THRESHOLD = 10.0

def detect_orb_similarity(image_path1, image_path2):
    """
    Detects similarity between two images using the ORB feature matching algorithm.

    Args:
        image_path1 (str): Path to the first image.
        image_path2 (str): Path to the second image.

    Returns:
        float: A similarity score between 0 and 100.
    """
    try:
        # 1. Load images in grayscale
        img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            print(f"Error: Could not read one or both images. Check paths:\n- {image_path1}\n- {image_path2}")
            return 0.0
            
    except Exception as e:
        print(f"Error loading images: {e}")
        return 0.0

    # 2. Initialize the ORB detector
    orb = cv2.ORB_create(nfeatures=N_FEATURES)

    # 3. Find keypoints and compute descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    
    if descriptors1 is None or descriptors2 is None:
        print("Warning: Could not compute descriptors for one or both images.")
        return 0.0

    # 4. Create a Brute-Force Matcher and find k-best matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # 5. Filter for "good" matches using Lowe's ratio test
    good_matches = []
    # Check if matches were found
    if len(matches) > 0 and len(matches[0]) == 2:
        for m, n in matches:
            if m.distance < GOOD_MATCH_THRESHOLD * n.distance:
                good_matches.append(m)

    # 6. Calculate similarity score
    min_keypoints = min(len(keypoints1), len(keypoints2))
    if min_keypoints == 0:
        return 0.0
        
    similarity_score = (len(good_matches) / min_keypoints) * 100
    
    return similarity_score

# --- Main script execution ---
if __name__ == "__main__":
    # 1. Check if the correct number of command-line arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python compare_images.py <path_to_image1> <path_to_image2>")
        sys.exit(1)

    # 2. Get image paths from arguments
    image1_path = sys.argv[1]
    image2_path = sys.argv[2]
    
    # 3. Calculate the similarity score
    score = detect_orb_similarity(image1_path, image2_path)

    # 4. Print the final results

    if score > SIMILARITY_THRESHOLD:
        print("SIMILAR")
    else:
        print("DIFFERENT")
