import os

def get_img_files(frames_dir):
    if not os.path.isdir(frames_dir):
        print(f"[Empty] frames_dir not found or not a directory: {frames_dir}")
        return []
    files = [f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    def numeric_key(name: str):
        stem = os.path.splitext(name)[0]
        return int(stem) if stem.isdigit() else stem
    files.sort(key=numeric_key)
    return [os.path.join(frames_dir, f) for f in files]