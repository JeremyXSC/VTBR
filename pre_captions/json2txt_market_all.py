import os
import json
import numpy as np

''' 
dir_json = 'predictions_market.json' #json存储的文件目录
dir_txt = '/test_label_json/'    #txt存储目录
if not os.path.exists(dir_txt):
    os.makedirs(dir_txt)
list_json = os.listdir(dir_json)
 
 
 
def json2txt(path_json,path_txt):    #可修改生成格式
    with open(path_json,'r') as path_json:
        jsonx=json.load(path_json)
        with open(path_txt,'w+') as ftxt:
            for shape in jsonx['shapes']:
                label = str(shape['label'])+' '
                xy=np.array(shape['points'])
                strxy = ''
                for m,n in xy:
                    m=int(m)
                    n=int(n)
                    strxy+=str(m)+' '+str(n)+' '
 
                label+=strxy
                ftxt.writelines(label+"\n")
 
 
for cnt,json_name in enumerate(list_json):
    print('cnt=%d,name=%s'%(cnt,json_name))
    path_json = dir_json + json_name
    path_txt = dir_txt + json_name.replace('.json','.txt')
    json2txt(path_json,path_txt)
'''
with open('predictions_market_212k_all.json','r') as path_json:
    predictions = json.load(path_json)
    print(len(predictions))
    with open('predictions_market_212k_all.txt','w+') as ftxt:
        for pred in predictions:
            if pred['image_id'].split('_')[0]=="-1" or pred['image_id'].split('_')[0]=="0000":continue
            else:ftxt.writelines("ImageName:"+pred['image_id']+".jpg, the caption is: "+pred['caption']+".\n")
    
