import os
import shutil
from PIL import Image
import imagehash
from tqdm import tqdm
import argparse

def organize_similar_images(source_dir, output_dir, threshold=5):
    """
    掃描資料夾並將相似圖片分組 (移動模式)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 取得所有圖片檔案路徑
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    all_files = [f for f in os.listdir(source_dir) if f.lower().endswith(valid_exts)]
    
    if not all_files:
        print(f"在 {source_dir} 中找不到支援的圖片檔案。")
        return

    # 2. 預先計算所有哈希值以提升速度
    print(f"正在計算 {len(all_files)} 張圖片的指紋...")
    hashes = {}
    for filename in tqdm(all_files, desc="計算哈希"):
        path = os.path.join(source_dir, filename)
        try:
            with Image.open(path) as img:
                # 使用 dhash (差異哈希)，在效能與準確度間平衡最佳
                hashes[filename] = imagehash.dhash(img)
        except Exception as e:
            print(f"跳過無法處理的檔案 {filename}: {e}")

    # 3. 進行比對並建立分組
    grouped_files = set()
    group_count = 0

    print("開始進行相似度比對與分類...")
    for i, file1 in enumerate(all_files):
        # 若已分類過或未成功產生哈希，則跳過
        if file1 in grouped_files or file1 not in hashes:
            continue
        
        current_group = [file1]
        
        # 與剩下的圖片進行比對
        for file2 in all_files[i+1:]:
            if file2 in grouped_files or file2 not in hashes:
                continue
            
            # 計算漢明距離 (Hamming Distance)
            distance = hashes[file1] - hashes[file2]
            
            if distance <= threshold:
                current_group.append(file2)
        
        # 4. 若找到相似圖片 (群組大於1)，則建立資料夾並移動
        if len(current_group) > 1:
            group_count += 1
            group_folder = os.path.join(output_dir, f"Group_{group_count}")
            os.makedirs(group_folder, exist_ok=True)
            
            for f in current_group:
                src_path = os.path.join(source_dir, f)
                dst_path = os.path.join(group_folder, f)
                
                # 改為移動檔案 (Move)
                shutil.move(src_path, dst_path)
                grouped_files.add(f)

    # 5. 處理未被分組的獨立圖片
    single_files = [f for f in all_files if f not in grouped_files and f in hashes]
    if single_files:
        print(f"\n正在將 {len(single_files)} 張未分組圖片移動至 'single' 資料夾...")
        single_folder = os.path.join(output_dir, "single")
        os.makedirs(single_folder, exist_ok=True)
        
        for f in single_files:
            src_path = os.path.join(source_dir, f)
            dst_path = os.path.join(single_folder, f)
            shutil.move(src_path, dst_path)
                
    print("\n處理完成！")
    print(f"▶ 共找到 {group_count} 組相似圖片。")
    print(f"▶ 有 {len(single_files)} 張圖片沒有相似項，已放入 single 資料夾。")
    print(f"▶ 檔案已全數移動至: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="圖片相似度自動分組工具")
    parser.add_argument("input", help="原始圖片資料夾路徑 (例如: ./images)")
    parser.add_argument("output", help="分類結果輸出的資料夾路徑 (例如: ./output)")
    parser.add_argument("-t", "--threshold", type=int, default=8, help="相似度閾值，預設為 8")
    
    args = parser.parse_args()
    
    print(f"▶ 輸入資料夾: {args.input}")
    print(f"▶ 輸出資料夾: {args.output}")
    print(f"▶ 相似度閾值: {args.threshold}\n")
    
    organize_similar_images(args.input, args.output, threshold=args.threshold)