#!/usr/bin/env python3
"""
Rename the first .txt file in each data/ subfolder to input.txt.

Usage: python auto_rename.py
"""
import os


def auto_rename_txt(data_folder="data"):
    for folder in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder)
        if not os.path.isdir(folder_path):
            continue
        if os.path.exists(os.path.join(folder_path, "input.txt")):
            print(f"✓ {folder}/input.txt already exists")
            continue
        txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
        if txt_files:
            old_path = os.path.join(folder_path, txt_files[0])
            new_path = os.path.join(folder_path, "input.txt")
            os.rename(old_path, new_path)
            print(f"✅ {folder}: {txt_files[0]} → input.txt")
        else:
            print(f"⚠️  {folder}: no .txt file found")


if __name__ == "__main__":
    auto_rename_txt()
    print("\nDone! Run: streamlit run app.py")
