import sys, os
path = os.path.dirname(__file__)
sys.path.append(path)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if os.path.exists(r'%s\predict_error.json'%path): #用于重复检测前，事先删除报错文件。
    os.remove(r'%s\predict_error.json'%path)
    
try:
    import time, traceback, copy, glob, cv2, shutil, json
    import numpy as np
    import pandas as pd
    from collections import Counter

    from nets.utils.data_treat import letterbox_image, label2rgb, label_colormap, draw_label, dense_crf
    from nets.utils.function import find_last
    from deeplabv3 import Deeplabv3
    from unet import Unet
    from pspnet import PSPnet
    from keras.layers import Lambda, Softmax, Reshape
    from warnings import simplefilter
    simplefilter(action = 'ignore', category = RuntimeWarning)
    
except (ModuleNotFoundError,ImportError) as reason:
    print(str(reason))
    print('扩展包未安装或安装不正确，请重新安装相应的扩展包！')
    with open(r'%s\predict_error.json'%path,'w') as f:
        f.write(traceback.format_exc())
        f.write('\n')
        f.write('occur time is %s'%(time.ctime()))
    sys.exit()

try:
    def get_info():
        filepath = json.load(open(r"%s\basicSet.json"%path, encoding= 'utf-8'))[0]#读取路径配置文件中的内容
        test_path = filepath['predict_img_path'] #训练图片保存位置
        result_path = filepath['result_path'] #训练图片保存位置
        log_path = filepath['log_path'] #模型文件位置
        classes = filepath['classes'] #模型的类别名称
        HEIGHT = filepath['img_height'] #图片的高度
        WIDTH = filepath['img_width'] #图片的宽度
        bone_name = filepath['bone_name'] #主干网络名称
        model_name = filepath['model_name'] #模型名称
        area_per_pixel = filepath['area_per_pixel'] #每个像素点代表的面积
        bottom_point = filepath['bottom_point'] #地图左下角的坐标点
        top_point = filepath['top_point'] #地图右上角的坐标点

        model_name = model_name.lower().replace(" ", "") #忽略大小写的敏感，并忽略字符间的空格，方便用户的输入
        bone_name = bone_name.lower().replace(" ", "")
        log_path = log_path.replace(" ", "")

        bottom_point = bottom_point.replace(" ", "")
        top_point = top_point.replace(" ", "")
        if len(bottom_point):
            bottom_point = tuple(map(float, bottom_point.split(","))) #将字符串格式转成元组格式
        if len(top_point):
            top_point = tuple(map(float, top_point.split(",")))

        if model_name not in ["deeplabv3", "unet", "pspnet"]: 
            raise ValueError('请输入正确的模型名称(大小写不敏感)，可选项为：["deeplabv3", "unet", "pspnet"]')
        if bone_name not in ["new_modelv1", "efficientnet_b0", "efficientnet_b2", "mobilenetv2", "mobilenetv3", "inceptionresnetv2"]:
            raise ValueError('请输入正确的主干网络名称(大小写不敏感)，可选项为：\
                            ["new_modelv1", "efficientnet_b0", "efficientnet_b2", "mobilenetv2", "mobilenetv3", "inceptionresnetv2"]')

        return test_path, result_path, log_path, classes, HEIGHT, WIDTH, bone_name, model_name, \
               area_per_pixel, bottom_point, top_point
    
    def predict(model_name, mask_save = False, combine_save = True, crf_treat = True, keep_path_struct = True): 
        '''
        参数说明：
        mask_save：设定是否单独保存预测结果
        combine_save：设定是否保存与原图的融合结果
        crf_treat：设定是否将预测结果进行CRF后处理---该处理目的是使得预测边缘平滑化，减少误判
        keep_path_struct：设定预测结果的保存，是否遵循原预测文件的架构。False代表统一存放到设定的文件夹下
        '''
        def real_pred(imgs):
            for nums, jpg in enumerate(imgs):
                img_name = os.path.basename(jpg)
                img_dir = os.path.dirname(jpg)
                if keep_path_struct: 
                    save_path = img_dir.replace(test_path, result_path) + "/"
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                else:
                    save_path = result_path + "/"
                img = cv2.imread(jpg)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #通道转换
                old_img = copy.deepcopy(img)
                orininal_h, orininal_w = old_img.shape[:2]
                img = letterbox_image(img, (HEIGHT, WIDTH)) #先处理成原始模型训练的尺寸大小
                img = img/255
                img = img.reshape(-1, HEIGHT, WIDTH, 3)

                pr = model.predict(img)[0]
                if crf_treat:
                    pr = pr.reshape((-1, HEIGHT, WIDTH, NCLASSES))
                    temp_img = letterbox_image(old_img, (HEIGHT, WIDTH)) #原图需要先转成训练时指定的图片大小
                    temp_img = temp_img[np.newaxis, ...] #需增加一个维度，方便输入CRF处理
                    pr = dense_crf(pr, img = temp_img, n_classes = NCLASSES) #增加CRF后处理
                pr = pr.reshape((HEIGHT, WIDTH, NCLASSES)).argmax(axis=-1) #将刚检测出来的图片，转成单通道的标签图片大小的数据格式
                pr = np.uint8(pr)
                pr = cv2.resize(pr, (orininal_w, orininal_h))
                for num in range(1, NCLASSES): #逐个将每个类别的像素值，计入各自的列表中
                    num_cal = Counter(pr[pr==num])[num] #Counter的结果类似一个字典，直接取其键值
                    areas[classes[num]].append(num_cal)
                if mask_save:
                    mask = label2rgb(pr, n_labels = len(classes), colormap = colors, mask_save = mask_save) #给mask上色, 注释掉则变单通道图
                    B, G, R = cv2.split(mask)
                    A = np.zeros(B.shape, dtype= B.dtype) #增加alpha通道，方便前端调用
                    A[B>0] = 255
                    mask = cv2.merge((B, G, R, A))
                    cv2.imwrite(save_path + img_name[:-4] +".png", mask)
                if combine_save:
                    image = draw_label(pr, old_img, classes, colormap = colors)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(save_path + img_name, image)
                print('文件夹%s的第%d张图片检测完毕'%(img_dir , nums + 1))

        if model_name == "deeplabv3":
            model = Deeplabv3(input_shape=(HEIGHT,WIDTH,3), classes=NCLASSES, bone_name = bone_name)
        elif model_name == 'unet':
            model = Unet(input_shape=(HEIGHT,WIDTH,3), classes=NCLASSES, bone_name = bone_name)
        elif model_name == 'pspnet':
            model = PSPnet(input_shape=(HEIGHT,WIDTH,3), classes=NCLASSES, bone_name = bone_name)
        log_name = os.path.basename(find_last(log_path))
        print('-'*100)
        print('Loading log_name of %s'%log_name)
        print('-'*100)
        model.load_weights(find_last(log_path), by_name=True)

        #需要根据类别数，来随机设定不同类别的mask对应的颜色
        colors = label_colormap(NCLASSES) #定义函数来确定不同类别的颜色，相对颜色识别度较高。
        total_area = 0 #初始化总面积
        areas = {} #设置空字典，用于存放每张图片各个类别的面积，并最终求和
        for num in range(1, NCLASSES): #先初始化每个类别，用于存放每张图得到的各类别像素点个数
            areas[classes[num]] = []
        print('开始检测...')
        for roots, _, _ in os.walk(test_path): #递归遍历多级子目录
            imgs = glob.glob(roots + "/*.jpg") #这里需注意，如果原图片是png格式的，需更改成.png
            real_pred(imgs)

        #统计各类别的面积
        f = open(result_path + '/area.txt', 'w')
        for num in range(1, NCLASSES):
            area = sum(areas[classes[num]])
            total_area +=area
            f.write('%s area is : %.2f m2 \n'%(classes[num], area * area_per_pixel))
        f.write('total area is : %.2f m2'%(total_area * area_per_pixel))
        f.close()
        t_end = time.time()
        time_total = pd.to_timedelta(t_end - t_start,unit = 's')
        print('%s_%s模型检测结束, 检测图片%s张, 结束时间为%s, 总耗时%s秒'%(bone_name, model_name, len(areas[classes[num]]), time.strftime('%Y-%m-%d %H:%M:%S'), time_total.round('s')))

    if __name__ =="__main__":
        t_start = time.time()
        print('开始执行检测，开始时间为: %s'%time.strftime('%Y-%m-%d %H:%M:%S'))        
        test_path, result_path, log_path, classes, HEIGHT, WIDTH, bone_name, model_name,\
        area_per_pixel, bottom_point, top_point = get_info()
        NCLASSES = len(classes)
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)            
        if os.listdir(result_path):
            shutil.rmtree(result_path, ignore_errors= True)  
        if not os.path.exists(result_path): #该行代码，确保前面的shutil.rmtree命令删除文件夹后仍旧保留空文件夹。
            os.makedirs(result_path)

        if len(os.listdir(test_path)): #如存在已有图片，直接检测
            print('使用已有图片进行检测...')
        elif len(bottom_point) and len(top_point): #必须要求坐标点非空
            from gog.get_pic import download_map
            print('开始进行图片下载...')
            test_path = path + "/gog/cache" #用于指向缓存文件夹，末尾不增加反斜杠，避免后面设置save_path时出错
            download_map(test_path, bottom_point, top_point) #下载地图
        else:
            raise ValueError('请提供正确的地图坐标点位(左下角和右上角的地图经纬度值)，或提供图片进行检测。')

        if not len(log_path):
            log_path = path + "/%s_%s_logs/"%(model_name, bone_name)
        if not os.path.exists(log_path): 
            tips_1 = "请事先训练模型，再进行检测。"
            raise FileNotFoundError(tips_1)
        predict(model_name, mask_save = False, combine_save = True, crf_treat = True, keep_path_struct = True)
#系统退出，程序运行人为中断，生成器等特殊问题，不考虑记录。
except Exception as reason:
    with open(r'%s\predict_error.json'%path, 'w') as f:
        f.write(traceback.format_exc())
        f.write('\n')
        f.write('occur time is %s'%(time.strftime('%Y-%m-%d %H:%M:%S')))
    print('Error occurring,reason is %s'%str(reason))