import fitz
import os

def pdf_pages_to_images_no_poppler(pdf_path, output_folder, start_page, end_page, dpi=150):
    doc = fitz.open(pdf_path)
    os.makedirs(output_folder, exist_ok=True)

    for page_num in range(start_page - 1, end_page):
        page = doc.load_page(page_num)
        zoom = dpi / 72  
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        output_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
        pix.save(output_path)
    
    print("Done!")

pdf_path = r"C:\Users\ce-sh\Desktop\divinaproportion00paci.pdf"
output_folder = r"C:\Users\ce-sh\Desktop\divinaproportion00paci"
pdf_pages_to_images_no_poppler(pdf_path, output_folder, start_page=1, end_page=135)