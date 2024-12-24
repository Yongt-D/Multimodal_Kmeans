from PIL import Image

# 打开 TIFF 文件并转换为 PNG
def convert_tiff_to_png(tiff_path, png_path):
    with Image.open(tiff_path) as img:
        img.save(png_path, 'PNG')

# 使用示例
tiff_image_path = 'E:/Phd1_homework/Data_analysis/test2.tif'
png_image_path = 'E:/Phd1_homework/Data_analysis/test2.png'
convert_tiff_to_png(tiff_image_path, png_image_path)
