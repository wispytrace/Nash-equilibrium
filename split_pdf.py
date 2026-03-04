import fitz  # PyMuPDF 的导入名称
import os

def pdf_to_images(pdf_path, output_dir, zoom=2.0):
    """
    将 PDF 的每一页转换为图片并保存。

    参数:
        pdf_path (str): 源 PDF 文件的路径。
        output_dir (str): 保存输出图片的文件夹路径。
        zoom (float): 缩放倍数。数值越大分辨率越高，图片越清晰。默认设为 2.0。
    """
    # 检查并创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # 打开 PDF 文档
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)
        print(f"正在处理: {pdf_path} (共 {total_pages} 页)")

        # 遍历所有页面
        for page_number in range(total_pages):
            page = pdf_document.load_page(page_number)
            
            # 设置矩阵以提升输出图片的分辨率
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix)
            
            # 构造输出文件名 (例如: page_1.png)
            output_file = os.path.join(output_dir, f"page_{page_number + 1}.png")
            
            # 保存图片
            pix.save(output_file)
            print(f"成功导出: {output_file}")

        pdf_document.close()
        print("🎉 PDF 拆分完毕！")

    except Exception as e:
        print(f"处理时发生错误: {e}")

# --- 测试运行 ---
if __name__ == "__main__":
    # 请将下面两个路径替换为您自己的实际路径
    source_pdf_file = "pdf_folder/pdf/25_Automatica_Resilient_fixed_time_distributed_Nash_equilibrium_seeking_algorithm__0228__ (4).pdf"  
    output_image_folder = "pdf_folder/images"
    
    pdf_to_images(source_pdf_file, output_image_folder)