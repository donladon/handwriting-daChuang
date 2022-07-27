import os
import numpy as np
import struct
import PIL.Image
 
train_data_dir = "HWDB1.1trn_gnt"
test_data_dir = "HWDB1.1tst_gnt"
 

# 读取图像和对应的汉字
def read_from_gnt_dir(gnt_dir=train_data_dir):
	
	def one_file(f):
		header_size = 10
		while True:
			header = np.fromfile(f, dtype='uint8', count=header_size)
			if not header.size: break
			sample_size = header[0] + (header[1]<<8) + (header[2]<<16) + (header[3]<<24)
			tagcode = header[5] + (header[4]<<8)
			width = header[6] + (header[7]<<8)
			height = header[8] + (header[9]<<8)
			if header_size + width*height != sample_size:
				break
			image = np.fromfile(f, dtype='uint8', count=width*height).reshape((height, width))
			yield image, tagcode 
 
	for file_name in os.listdir(gnt_dir): # 读取所有文件
		if file_name.endswith('.gnt'):
			file_path = os.path.join(gnt_dir, file_name)
			with open(file_path, 'rb') as f:
				for image, tagcode in one_file(f): #在这调用one_file函数
					yield image, tagcode #迭代器





# 统计样本数
train_counter = 0
test_counter = 0


for image, tagcode in read_from_gnt_dir(gnt_dir=train_data_dir):  #统计train样本数
	tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312') #tagcode_unicode编码为汉字

	# 提取一些图像
	if train_counter < 1000:
		im = PIL.Image.fromarray(image)
		im.convert('RGB').save('png/' + tagcode_unicode + str(train_counter) + '.png') #保存图像为XXX.png

	
	train_counter += 1 

for image, tagcode in read_from_gnt_dir(gnt_dir=test_data_dir):   #统计test样本数
	tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
	test_counter += 1
 
# 样本数
print(train_counter, test_counter)