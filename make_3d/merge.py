import os
from PIL import Image
from io import BytesIO


file_list = os.listdir('svideo_result_left')
#print(file_list)

for fname in file_list:
    images = []
    img = Image.open('svideo_result_left/'+fname)
    images.append(img)
    sname = fname.split('_')
    img2 = Image.open('svideo_result_right/'+ sname[0]+'_right_'+sname[2])
    images.append(img2)
    new_img = Image.new("RGB",(384*2,160),'white')
    width = 0
    for index in images:
        new_img.paste(index,(width,0))
        width += 384
    new_img.save('svideo_result/'+sname[0]+'_result_'+sname[2])
    print(new_img)
