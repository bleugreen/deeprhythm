#!/usr/bin/env python3
"""Test script to verify model_path parameter behavior."""

import os
import sys
from deeprhythm.model.predictor import DeepRhythmPredictor

def test_default_behavior():
    """Test that model_path=None auto-downloads."""
    print("Test 1: Default behavior (auto-download)...")
    try:
        predictor = DeepRhythmPredictor(quiet=True)
        print(f"  ✓ Success! Model loaded from: {predictor.model_path}")
        assert os.path.isfile(predictor.model_path), "Model file should exist"
        assert ".local/share/deeprhythm" in predictor.model_path or "deeprhythm" in predictor.model_path, \
            "Should use default location"
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

def test_invalid_path():
    """Test that invalid path raises FileNotFoundError."""
    print("\nTest 2: Invalid custom path...")
    try:
        predictor = DeepRhythmPredictor(model_path="/invalid/path/model.pth", quiet=True)
        print(f"  ✗ Failed: Should have raised FileNotFoundError")
        return False
    except FileNotFoundError as e:
        print(f"  ✓ Success! Raised FileNotFoundError as expected")
        print(f"     Error message: {str(e)[:80]}...")
        return True
    except Exception as e:
        print(f"  ✗ Failed with unexpected error: {e}")
        return False

def test_valid_custom_path():
    """Test that valid custom path is used."""
    print("\nTest 3: Valid custom path...")
    try:
        # First get the default model path
        temp_predictor = DeepRhythmPredictor(quiet=True)
        default_path = temp_predictor.model_path
        
        # Now use it as a custom path
        predictor = DeepRhythmPredictor(model_path=default_path, quiet=True)
        print(f"  ✓ Success! Model loaded from custom path: {predictor.model_path}")
        assert predictor.model_path == default_path, "Should use the provided path"
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("Testing model_path parameter behavior")
    print("="*60)
    
    results = []
    results.append(test_default_behavior())
    results.append(test_invalid_path())
    results.append(test_valid_custom_path())
    
    print("\n" + "="*60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("="*60)
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
