import os
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# ================= CONFIGURATION =================

current_script_path = Path(__file__).resolve()
parent_dir = current_script_path.parents[2]
cwd = os.path.join(parent_dir, 'Task2_test_data')
IMAGE_PATH = os.path.join(cwd, '/images') # origanizer's data
SAVE_PATH = os.path.join(cwd, 'processed_pngs_mix')
MAX_SIZE =2048


# ===========================================

def batch_convert_to_png():
    # 确保保存路径存在
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        print(f"Create a directory: {SAVE_PATH}")

    # 获取所有图片文件
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')
    files = [f for f in os.listdir(IMAGE_PATH) if f.lower().endswith(valid_extensions)]

    print(f" {len(files)} images，processing...")

    for filename in tqdm(files, desc="Processing Images"):
        try:
            img_path = os.path.join(IMAGE_PATH, filename)
            with Image.open(img_path) as img:

                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGBA")
                else:
                    img = img.convert("RGB")

                width, height = img.size
                if max(width, height) > MAX_SIZE:
                    if width > height:
                        new_width = MAX_SIZE
                        new_height = int(MAX_SIZE * height / width)
                    else:
                        new_height = MAX_SIZE
                        new_width = int(MAX_SIZE * width / height)

                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                name_without_ext = os.path.splitext(filename)[0]
                save_file_path = os.path.join(SAVE_PATH, f"{name_without_ext}.png")


                img.save(save_file_path, "PNG", optimize=True)

        except Exception as e:
            print(f"\nSkip: {filename}: {e}")

    print(f"\nAll done, saved at: {SAVE_PATH}")


if __name__ == "__main__":
    batch_convert_to_png()