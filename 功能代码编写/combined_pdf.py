import os
# import fitz  # PyMuPDF
from PyPDF4 import PdfFileReader, PdfFileWriter


def merge_pdfs(src_folder, output_pdf):
    """
    合并指定文件夹中的所有PDF文件到一个PDF文件中。

    :param src_folder: 包含PDF文件的源文件夹路径
    :param output_pdf: 输出合并后PDF的文件路径
    """
    # 获取文件夹中所有PDF文件的列表，并按文件名排序
    pdf_files = [os.path.join(src_folder, f) for f in os.listdir(src_folder) if f.endswith('.pdf')]
    pdf_files.sort()

    # 初始化一个PdfFileWriter对象，用于写入合并后的PDF内容
    pdf_writer = PdfFileWriter()

    for pdf_file in pdf_files:
        # 打开每个PDF文件并读取内容
        with open(pdf_file, 'rb') as file:
            pdf_reader = PdfFileReader(file)
            # 将当前PDF文件的所有页面添加到writer对象中
            for page_num in range(pdf_reader.getNumPages()):
                pdf_writer.addPage(pdf_reader.getPage(page_num))

    # 写入到最终的PDF文件
    with open(output_pdf, 'wb') as out:
        pdf_writer.write(out)

    print(f"PDF文件已成功合并至：{output_pdf}")

    # filenames = ['Python数据科学速查表 - Jupyter Notebook.pdf', 'Python数据科学速查表 - Matplotlib 绘图.pdf',
    #              'Python数据科学速查表 - Numpy 基础.pdf', 'Python数据科学速查表 - Pandas 基础.pdf',
    #              'Python数据科学速查表 - Pandas 进阶.pdf', 'Python数据科学速查表 - Python 基础.pdf']
    # merger = PyPDF4.PdfFileMerger()
    #
    # for filename in pdf_files:
    #     merger.append(PyPDF4.PdfFileReader(filename))
    # merger.write('pdf/电力电子运控.pdf')

    # # 创建一个新的PDF文档用于合并
    # merged_pdf = fitz.open()
    #
    # for pdf_file in pdf_files:
    #     # 打开每个PDF文件
    #     with fitz.open(pdf_file) as pdf_doc:
    #         # 将当前文档的所有页面添加到合并的PDF文档中
    #         for page in pdf_doc:
    #             merged_pdf.insert_page(-1, page)
    #
    # # 保存合并后的PDF文件
    # merged_pdf.save(output_pdf)
    # merged_pdf.close()
    # print(f"PDF文件已成功合并至：{output_pdf}")


# 使用示例
src_folder = 'pdf'  # 指定包含PDF文件的文件夹路径
output_pdf = 'pdf/电力电子运控.pdf'  # 指定输出的合并PDF文件名
merge_pdfs(src_folder, output_pdf)