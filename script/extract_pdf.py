from pypdf import PdfReader, PdfWriter

def extract_first_n_pages(input_path, output_path, n=8):
    try:
        # 读取原始 PDF 文件
        reader = PdfReader(input_path)
        writer = PdfWriter()

        # 获取 PDF 的总页数
        total_pages = len(reader.pages)
        print(f"原文件总页数: {total_pages}")

        # 确定需要提取的实际页数（防止总页数不足 n 页）
        pages_to_extract = min(n, total_pages)

        # 逐页提取并添加到写入器中
        for i in range(pages_to_extract):
            page = reader.pages[i]
            writer.add_page(page)

        # 将提取的页面保存到新的 PDF 文件中
        with open(output_path, "wb") as output_file:
            writer.write(output_file)
            
        print(f"成功提取前 {pages_to_extract} 页，并保存至: {output_path}")

    except FileNotFoundError:
        print(f"错误：找不到文件 '{input_path}'，请检查文件路径是否正确。")
    except Exception as e:
        print(f"发生了一个错误: {e}")

# === 使用示例 ===
if __name__ == "__main__":
    # 请将下面这两个路径替换为你实际的文件路径
    input_pdf = "CityU_A_study_of_distributed_Nash_equilibrium_seeking_algorithm_for_non_cooperative_game_with_fixed_time_convergence.pdf"   # 原始 PDF 路径
    output_pdf = "[binghao]_Thesis_Abstract.pdf"  # 提取后保存的 PDF 路径
    
    extract_first_n_pages(input_pdf, output_pdf, n=8)