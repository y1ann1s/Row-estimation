import numpy as np
from urllib.parse import urlparse, urljoin
from urllib.request import urlopen

def is_valid_url(url):
    parsed_url = urlparse(url)
    if parsed_url.scheme not in ("http", "https"):
        return False
    safe_domains = ["youtube.com", "vimeo.com", "example.com"]
    if parsed_url.netloc not in safe_domains:
        return False
    return True

def get_content_type(url):
    try:
        response = urlopen(url)
        content_type = response.info().get_content_type()
        return content_type
    except Exception as e:
        print(f"Failed to get content type for URL {url}: {e}")
        return None

def calculate_angle(pt1, pt2, pt3):
    vector1 = np.array([pt1[0] - pt2[0], pt1[1] - pt2[1]])
    vector2 = np.array([pt3[0] - pt2[0], pt3[1] - pt2[1]])
    angle_rad = np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
    angle_deg = np.degrees(angle_rad)
    return angle_deg
