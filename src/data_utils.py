from pathlib import Path

def get_train_paths(data_path: Path):
    data_paths = []
    for class_dir in data_path.iterdir():
        class_id = class_dir.parts[-1]
        for img in class_dir.iterdir():
            data_paths.append({"label": class_id, "image": img})

    return data_paths
