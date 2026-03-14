"""Merge training data files"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.log import wprint


def find_data_files(data_dir: str, patterns: List[str] = None) -> List[str]:
    """Find all data files matching patterns"""
    import glob

    if patterns is None:
        patterns = ["selfplay_*.pt", "online_*.pt"]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(data_dir, pattern)))

    return sorted(set(files))


def get_file_info(filepath: str) -> dict:
    """Get file information without loading full data"""
    stat = os.stat(filepath)
    filename = os.path.basename(filepath)

    # Determine file type
    if filename.startswith("selfplay"):
        file_type = "selfplay"
    elif filename.startswith("online"):
        file_type = "online"
    else:
        file_type = "unknown"

    return {
        "path": filepath,
        "filename": filename,
        "type": file_type,
        "size_mb": stat.st_size / (1024 * 1024),
        "mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }


def merge_data_files(
    input_files: List[str],
    output_path: str,
    deduplicate: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Merge multiple data files into one.

    Args:
        input_files: List of input file paths
        output_path: Output file path
        deduplicate: Whether to deduplicate boards (slower)
        dry_run: Only show what would be done

    Returns:
        Statistics about the merge operation
    """
    if not input_files:
        wprint("没有找到输入文件")
        return {"status": "error", "message": "No input files"}

    wprint("=" * 60)
    wprint("数据合并")
    wprint("=" * 60)

    # Show input files
    wprint(f"\n找到 {len(input_files)} 个输入文件:")
    wprint("-" * 60)

    total_size = 0
    for f in input_files:
        info = get_file_info(f)
        total_size += info["size_mb"]
        wprint(f"  [{info['type']:8}] {info['filename']}")
        wprint(f"           大小: {info['size_mb']:.1f} MB, 修改: {info['mtime']}")

    wprint("-" * 60)
    wprint(f"总大小: {total_size:.1f} MB")

    if dry_run:
        wprint(f"\n[预演] 将合并到: {output_path}")
        return {"status": "dry_run", "files": len(input_files), "total_size_mb": total_size}

    # Load and merge data
    wprint(f"\n开始合并...")

    all_boards = []
    all_policies = []
    all_values = []
    all_game_ids = []

    for i, filepath in enumerate(input_files, 1):
        info = get_file_info(filepath)
        wprint(f"  [{i}/{len(input_files)}] 加载 {info['filename']}...")

        try:
            data = torch.load(filepath, weights_only=False)
            num_samples = len(data["boards"])
            file_game_ids = data.get("game_ids")
            if file_game_ids is None:
                file_game_ids = [None] * num_samples

            all_boards.extend(data["boards"])
            all_policies.extend(data["policies"])
            all_values.extend(data["values"])
            all_game_ids.extend(file_game_ids)

            wprint(f"           已加载 {num_samples} 个样本")
        except Exception as e:
            wprint(f"           错误: {e}")
            continue

    if not all_boards:
        wprint("没有成功加载任何数据!")
        return {"status": "error", "message": "No data loaded"}

    # Deduplicate if requested
    if deduplicate:
        wprint("\n去重中...")
        original_count = len(all_boards)

        # Use board hash for deduplication
        seen = set()
        unique_boards = []
        unique_policies = []
        unique_values = []
        unique_game_ids = []

        for board, policy, value, game_id in zip(
            all_boards, all_policies, all_values, all_game_ids
        ):
            # Create hash from board tensor
            if isinstance(board, torch.Tensor):
                board_hash = hash(board.contiguous().numpy().tobytes())
            else:
                board_hash = hash(str(board))

            if board_hash not in seen:
                seen.add(board_hash)
                unique_boards.append(board)
                unique_policies.append(policy)
                unique_values.append(value)
                unique_game_ids.append(game_id)

        all_boards = unique_boards
        all_policies = unique_policies
        all_values = unique_values
        all_game_ids = unique_game_ids

        wprint(f"  去重: {original_count} -> {len(all_boards)} (移除 {original_count - len(all_boards)} 个重复)")

    # Convert to tensors
    wprint("\n转换为张量...")
    boards_tensor = torch.stack(
        [b if isinstance(b, torch.Tensor) else torch.from_numpy(b) for b in all_boards]
    )
    policies_tensor = torch.stack(
        [p if isinstance(p, torch.Tensor) else torch.from_numpy(p) for p in all_policies]
    )
    values_tensor = torch.stack(
        [
            v.to(dtype=torch.float32) if isinstance(v, torch.Tensor)
            else torch.tensor(v, dtype=torch.float32)
            for v in all_values
        ]
    )

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Save merged data
    wprint(f"\n保存到: {output_path}")
    merged_data = {
        "boards": boards_tensor,
        "policies": policies_tensor,
        "values": values_tensor,
        "game_ids": all_game_ids,
    }

    torch.save(merged_data, output_path)

    output_size = os.path.getsize(output_path) / (1024 * 1024)

    wprint("\n" + "=" * 60)
    wprint("合并完成!")
    wprint(f"  输入文件: {len(input_files)} 个")
    wprint(f"  总样本数: {len(all_boards)} 个")
    wprint(f"  输出大小: {output_size:.1f} MB")
    wprint(f"  输出路径: {output_path}")
    wprint("=" * 60)

    return {
        "status": "success",
        "input_files": len(input_files),
        "total_samples": len(all_boards),
        "output_size_mb": output_size,
        "output_path": output_path,
    }


def list_data_files(data_dir: str) -> None:
    """List all data files with details"""
    files = find_data_files(data_dir)

    if not files:
        wprint(f"在 {data_dir} 中未找到数据文件")
        return

    wprint("=" * 60)
    wprint(f"数据文件列表: {data_dir}")
    wprint("=" * 60)

    total_size = 0
    total_samples = 0

    for f in files:
        info = get_file_info(f)
        total_size += info["size_mb"]

        # Load to get sample count
        try:
            data = torch.load(f, weights_only=False)
            num_samples = len(data["boards"])
            total_samples += num_samples
        except:
            num_samples = "?"

        wprint(f"\n  [{info['type']:8}] {info['filename']}")
        wprint(f"           样本数: {num_samples}, 大小: {info['size_mb']:.1f} MB")
        wprint(f"           修改时间: {info['mtime']}")

    wprint("\n" + "-" * 60)
    wprint(f"总计: {len(files)} 个文件, {total_samples} 个样本, {total_size:.1f} MB")
    wprint("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="合并训练数据文件")

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # merge command
    merge_parser = subparsers.add_parser("merge", help="合并数据文件")
    merge_parser.add_argument("--data", type=str, default="data", help="数据目录")
    merge_parser.add_argument(
        "--output", type=str, default=None, help="输出文件路径 (默认: data/merged_YYYYMMDD_HHMMSS.pt)"
    )
    merge_parser.add_argument(
        "--patterns", type=str, nargs="+", default=["selfplay_*.pt", "online_*.pt"], help="文件匹配模式"
    )
    merge_parser.add_argument("--dedup", action="store_true", help="去重")
    merge_parser.add_argument("--dry-run", action="store_true", help="仅预演，不实际合并")

    # list command
    list_parser = subparsers.add_parser("list", help="列出数据文件")
    list_parser.add_argument("--data", type=str, default="data", help="数据目录")

    # clean command
    clean_parser = subparsers.add_parser("clean", help="清理旧数据文件 (保留最新 N 个)")
    clean_parser.add_argument("--data", type=str, default="data", help="数据目录")
    clean_parser.add_argument("--keep", type=int, default=3, help="保留文件数")
    clean_parser.add_argument("--type", type=str, default="selfplay", choices=["selfplay", "online", "all"])
    clean_parser.add_argument("--dry-run", action="store_true", help="仅预演")

    args = parser.parse_args()

    if args.command == "merge":
        # Find files
        files = find_data_files(args.data, args.patterns)

        # Generate output path if not specified
        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = os.path.join(args.data, f"merged_{timestamp}.pt")

        merge_data_files(files, args.output, deduplicate=args.dedup, dry_run=args.dry_run)

    elif args.command == "list":
        list_data_files(args.data)

    elif args.command == "clean":
        files = find_data_files(args.data)

        # Filter by type
        if args.type != "all":
            files = [f for f in files if os.path.basename(f).startswith(args.type)]

        if len(files) <= args.keep:
            wprint(f"只有 {len(files)} 个文件，无需清理")
            return

        # Sort by modification time (newest first)
        files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

        keep_files = files[: args.keep]
        remove_files = files[args.keep :]

        wprint(f"保留 {len(keep_files)} 个最新文件:")
        for f in keep_files:
            wprint(f"  [保留] {os.path.basename(f)}")

        wprint(f"\n将删除 {len(remove_files)} 个旧文件:")
        for f in remove_files:
            wprint(f"  [删除] {os.path.basename(f)}")

        if not args.dry_run:
            confirm = input("\n确认删除? (y/N): ")
            if confirm.lower() == "y":
                for f in remove_files:
                    os.remove(f)
                    wprint(f"  已删除: {os.path.basename(f)}")
            else:
                wprint("取消删除")
        else:
            wprint("\n[预演] 未实际删除文件")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
