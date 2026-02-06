import pytest
from PIL import Image
import numpy as np
from src.utils.vision import VisionUtils

def test_compute_similarity_identical():
    img = Image.new('RGB', (100, 100), color='red')
    score = VisionUtils.compute_similarity(img, img)
    assert score > 0.99

def test_compute_similarity_different():
    img1 = Image.new('RGB', (100, 100), color='white')
    img2 = Image.new('RGB', (100, 100), color='black')
    score = VisionUtils.compute_similarity(img1, img2)
    assert score < 0.5

def test_mark_action():
    img = Image.new('RGB', (100, 100), color='white')
    coords = [50, 50]
    marked = VisionUtils.mark_action(img, coords)
    
    orig_arr = np.array(img)
    marked_arr = np.array(marked)
    assert not np.array_equal(orig_arr, marked_arr)