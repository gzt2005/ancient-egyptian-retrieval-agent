from pathlib import Path

base_dir = Path(r"C:\Users\GE ZITONG\Desktop\egypt_agent_project\data_raw\aes\files")

folders_to_check = [
    base_dir / "aes",
    base_dir / "relANNIS"
]

for folder in folders_to_check:
    print("\n" + "=" * 80)
    print("正在检查文件夹：", folder)
    print("是否存在：", folder.exists())
    print("是否为文件夹：", folder.is_dir())

    if not folder.exists():
        continue

    items = list(folder.glob("*"))
    print("一级文件/文件夹数量：", len(items))

    print("\n前 80 个一级内容：")
    for item in items[:80]:
        print(item.name)
        print("  是文件夹吗：", item.is_dir())
        print("  是文件吗：", item.is_file())
        print("  后缀：", item.suffix)
        print("  完整路径：", item)
        print()

    # 额外统计所有后缀
    all_files = [p for p in folder.rglob("*") if p.is_file()]
    suffix_count = {}

    for f in all_files:
        suffix = f.suffix if f.suffix else "[无后缀]"
        suffix_count[suffix] = suffix_count.get(suffix, 0) + 1

    print("所有文件数量：", len(all_files))
    print("后缀统计：")
    for suffix, count in sorted(suffix_count.items(), key=lambda x: x[0]):
        print(" ", suffix, ":", count)