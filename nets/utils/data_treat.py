import numpy as np, os, cv2, shutil, json, copy, glob, io
from labelme import utils
from PIL import Image
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf

class PicAug(): #数据增强手段
    #图片翻转处理(3种)
    def filp(self, img, filename, filepath, ext):
        img_hor = cv2.flip(img, 1) #水平翻转
        img_ver = cv2.flip(img, 0) #垂直翻转
        img_filp = cv2.flip(img, -1) #水平+垂直翻转
        cv2.imwrite(filepath + '%s_filp_hor_conv%s'%(filename, ext), img_hor)
        cv2.imwrite(filepath + '%s_filp_ver_conv%s'%(filename, ext), img_ver)
        cv2.imwrite(filepath + '%s_filp_conv%s'%(filename, ext), img_filp)
    #增加椒盐噪声
    def SaltAndPepperNoise(self, img, filename, filepath, ext):
        SP_NoiseImg = img
        percentage = 0.001
        #这里要求图片本身就是灰度图或是黑白图
        SP_NoiseNum = int(percentage * img.shape[0] * img.shape[1]) #设置椒盐噪声的面积占比
        for _ in range(SP_NoiseNum):
            randX = np.random.randint(0, img.shape[0] - 1) #-1是因为起始值为0
            randY = np.random.randint(0, img.shape[1] - 1)
            if np.random.randint(0, 1) == 0:
                SP_NoiseImg[randX, randY] = 0 #0为胡椒噪声，其实就是随机设置某些点的像素值为0，黑色点
            else:
                SP_NoiseImg[randX, randY] = 255 #1为盐粒噪声，随机设置某些点的像素值为255，白色点
        cv2.imwrite(filepath + '%s_SPnoise_conv%s'%(filename, ext), SP_NoiseImg)
    #增加高斯噪声
    def AddGausisnNoise(self, img, filename, filepath, ext):
        G_NoiseImg = img
        percentage = 0.001
        G_NoiseNum = int(percentage * img.shape[0] * img.shape[1]) #设置高斯噪声的面积占比
        for _ in range(G_NoiseNum):
            randX = np.random.randint(0, G_NoiseImg.shape[0])
            randY = np.random.randint(0, G_NoiseImg.shape[1])
            G_NoiseImg[randX][randY] = 255 #将随机点设置为255，白色点
        cv2.imwrite(filepath + '%s_GSnoise_conv%s'%(filename, ext), G_NoiseImg)
    #增加同态滤波
    def homo_transfer(self, img, filename, filepath, ext):
        def homomorphic_filter(label, d0 = 5, r1 = 0.5, rh=2, c=4, h=2.0, l=0.5):
            gray = label.copy()
            if len(label.shape) > 2:
                gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
            gray = np.float64(gray)
            row, col = gray.shape
            gray_fft = np.fft.fft2(gray)
            gray_fftshift = np.fft.fftshift(gray_fft)
            dst_fftshift = np.zeros_like(gray_fftshift)
            M,N = np.meshgrid(np.arange(-col // 2,col // 2),np.arange(-row//2, row//2))
            D = np.sqrt(M ** 2 + N ** 2)
            Z = (rh - r1) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + r1
            dst_fftshift = Z * gray_fftshift
            dst_fftshift = (h - l) * dst_fftshift + l
            dst_ifftshift = np.fft.ifftshift(dst_fftshift)
            dst_ifft = np.fft.ifft2(dst_ifftshift)
            dst = np.real(dst_ifft)
            dst = np.uint8(np.clip(dst, 0, 255))
            return dst
        #同态滤波处理的是灰度图，需要对每个通道单独处理，再进行合并，方能得到三通道的增强图
        if img.shape[-1] == 3:
            dst0 = homomorphic_filter(img[:, :, 0])
            dst1 = homomorphic_filter(img[:, :, 1])
            dst2 = homomorphic_filter(img[:, :, 2])
            dst = np.zeros_like(img, dtype = np.uint8)
            dst[:, :, 0] = dst0
            dst[:, :, 1] = dst1
            dst[:, :, 2] = dst2
        else:
            dst = homomorphic_filter(img)
        cv2.imwrite(filepath + '%s_homofilter_conv%s'%(filename, ext), dst)
    #增加模糊处理(4种)
    def blur(self, img, filename, filepath, ext):
        blur_img = img.copy()
        img_mean_blur = cv2.blur(blur_img, (10, 10)) #均值模糊
        img_median_blur = cv2.medianBlur(blur_img, 5) #中值模糊
        img_GS_blur = cv2.GaussianBlur(blur_img, (0, 0), 3) #高斯模糊
        img_BI_blur = cv2.bilateralFilter(blur_img, 0 ,30, 10) #双边模糊
        cv2.imwrite(filepath + '%s_meanblur_conv%s'%(filename, ext), img_mean_blur)
        cv2.imwrite(filepath + '%s_medianblur_conv%s'%(filename, ext), img_median_blur)
        cv2.imwrite(filepath + '%s_GSblur_conv%s'%(filename, ext), img_GS_blur)
        cv2.imwrite(filepath + '%s_BIblur_conv%s'%(filename, ext), img_BI_blur)
    
    #通道互换
    def channel_change(self, img, filename, filepath, ext):
        img_channel = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(filepath + '%s_channel_conv%s'%(filename, ext), img_channel)

    #线性增强，增强对比度
    def linear_conv(self, img, filename, filepath, ext):
        out = 3.0 * img
        out[out>255] = 255 #需注意此时的out为np.array数组格式
        out = np.around(out) #对数值取整（四舍五入）
        out = out.astype(np.uint8)
        cv2.imwrite(filepath + '%s_linear_conv%s'%(filename, ext), out)

    #正规化视图
    def norm_conv(self, img, filename, filepath, ext):
        out =np.zeros(img.shape, np.uint8)
        cv2.normalize(img, out, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imwrite(filepath + '%s_norm_conv%s'%(filename, ext), out)

    #伽马变换（将灰度值归一化，然后对每个像素值进行幂运算）
    def gamma_conv(self, img, filename, filepath, ext):
        img_norm = img / 255.0
        gamma = 0.4
        out = np.power(img_norm, gamma)
        out = out * 255 #由于幂运算后的值为浮点数（介于0, 1之间），需要将其乘255后方能正常保存图片
        cv2.imwrite(filepath + '%s_gamma_conv%s'%(filename, ext), out)
                        
def img_augment(img_path, label_path):
    for enhance_dir in [img_path, label_path]: #对原始图片和label图片进行数据增强
        for filename in os.listdir(enhance_dir): #数据增强
            img = cv2.imread(enhance_dir + filename, cv2.IMREAD_UNCHANGED)
            imgname = filename[:-4]
            ext = filename[-4:]
            PA = PicAug() #考虑到图片并非都是正方形，因此不进行旋转处理，否则标签图片转换比较复杂。
            PA.filp(img, imgname, enhance_dir, ext) #翻转，包含水平、垂直和180度翻转
            PA.SaltAndPepperNoise(img, imgname, enhance_dir, ext) #增加椒盐噪声
            PA.AddGausisnNoise(img, imgname, enhance_dir, ext) #增加高斯噪声
            PA.homo_transfer(img, imgname, enhance_dir, ext) #增加同态滤波
            PA.blur(img, imgname, enhance_dir, ext) #增加均值模糊
            PA.channel_change(img, imgname, enhance_dir, ext) #增加通道互换
            PA.linear_conv(img, imgname, enhance_dir, ext) #线性转换
            PA.norm_conv(img, imgname, enhance_dir, ext) #正规化转换
            PA.gamma_conv(img, imgname, enhance_dir, ext) #gama转换

            #将原始的label图片，替换增噪、同态滤波、模糊化、通道互换等产生的label图片，确保label图片的正确性
            for each_file in os.listdir(label_path):
                origin_name = each_file.split("_")[0]
                if 'noise_' in each_file or 'homofilter_' in each_file or 'blur_' in each_file or \
                    'channel_' in each_file or 'linear_' in each_file or 'norm_' in each_file or \
                    'gamma_' in each_file:
                    shutil.copyfile(label_path + origin_name + ".png", label_path + each_file)

def rename(filepath): #进行文件统一重命名，方便后续的数据增强等处理
    for each_file in os.listdir(filepath):
        if "_" in each_file:
            newname = each_file.replace("_", "-") #此处进行符号替换，以免导致后期数据增强出错
            os.rename(filepath + each_file, filepath + newname)
                    
def json_to_label(json_path, label_path, classes): #json文件转换
    NCLASSES = len(classes)
    NAME_LABEL_MAP = dict(zip(classes, range(NCLASSES))) #建立类别名称与像素值的对应关系
    filelist = os.listdir(json_path)
    for i in range(len(filelist)):
        path = os.path.join(json_path, filelist[i])
        filename = filelist[i][:-5] #去除".json"，剩余文件名
        if os.path.isfile(path):
            data = json.load(open(path))
            img = utils.image.img_b64_to_arr(data['imageData'])
            lbl =utils.shape.shapes_to_label(img.shape, data['shapes'],NAME_LABEL_MAP)
            lbl_tmp = copy.copy(lbl)
            for key_name in NAME_LABEL_MAP: #这里的代码与源码不一致，仍旧会确保像素值与类别一一对应
                old_lbl_val = NAME_LABEL_MAP[key_name]
                new_lbl_val = NAME_LABEL_MAP[key_name]
                lbl_tmp[lbl == old_lbl_val] = new_lbl_val
            lbl = np.array(lbl_tmp, dtype=np.int8)
            cv2.imwrite(label_path + filename + ".png",lbl)
    print('Json文件转label文件已结束!')

def get_data(img_path): #进行数据集切分
    images = os.listdir(img_path)
    np.random.seed(1) #设定随机选择数据集，以便对比训练结果
    np.random.shuffle(images) #打乱数据集
    num_train = int(len(images)*0.9) # 90%用于训练，10%用于验证。
    num_val = len(images) - num_train
    train_data = images[:num_train] #获取训练集图片的名称，以便传入数据集制作函数，生成img和label的数据集
    val_data = images[num_train:]
    return train_data, val_data, num_train, num_val

def letterbox_image(image, size):#按照设定的尺寸，统一输入图片的大小
    ih, iw = image.shape[:2]
    h, w = size
    scale = min(w/iw, h/ih)
    nh = int(ih*scale)
    nw = int(iw*scale)
    image = cv2.resize(image, (nw,nh), interpolation = cv2.INTER_CUBIC)#注意resize方法，是宽在前，高在后。
    new_image = np.zeros((h, w, 3),dtype = np.uint8)
    new_h = (h - nh)//2
    new_w = (w - nw)//2
    new_image[new_h : (new_h + nh), new_w : (new_w + nw)] = image
    return new_image  #返回np.array    

def image_treat(img_path, label_path, json_path, classes, augment = True): #默认进行数据增强
    '''
    图片处理过程：
    1. 先进行判断，是否label图片已经存在与否。如label图片数量为空，则进行json文件转换。json文件也为空，则报错。
    2. json文件转换出来后，如要求数据增强，则直接进行重命名 + 增强(转换前，已经事先判断过json文件是否和图片一致)。
    3. 如已直接提供label图片，则事先检查label和img数量和名称是否一致。
        1) 如要求数据增强：
        A) 如label和img文件一致，查看文件名称中是否含有增强标记。
            a）如含有增强标记，说明已经增强完成，直接进行训练。
            b) 如不含有增强标记，则说明没有增强过。进行重命名处理，并增强。
        B) 如label和img文件不一致，查看文件名称中是否含有增强标记。
            a）文件中含有增强标记，说明增强中断，则删除已增强文件，重新增强。
            b) 文件中不含有增强标记，则报错。
        2). 如不要求数据增强，且label和img一致，则直接训练。
    '''
    if not os.path.exists(img_path) or not len(os.listdir(img_path)):
        raise FileNotFoundError('未找到图片文件，请确认img_path设置无误')
    if len(glob.glob(label_path + "*.png")) == 0:
        if not os.path.exists(json_path):
            raise FileNotFoundError('未找到json文件，请确认json_path或label_path设置无误！')
        if len(json_path):
            json_files = sorted([i[:-5] for i in os.listdir(json_path)]) #将json和image文件名称进行排序
            image_files = sorted([i[:-4] for i in os.listdir(img_path)])
            if json_files == image_files: #事先判断json文件，是否原始图片一致，以防后面数据不匹配。
                json_to_label(json_path, label_path, classes) #先将标注好的json文件转成label图片
                if augment:
                    rename(img_path) #json文件不进行改名。因数据增强仅针对png和jpg图片处理。
                    rename(label_path)
                    print('开始进行数据增强，请耐心等待一段时间，增强完成将会提示...')
                    img_augment(img_path, label_path)
                    print('数据已增强完毕！')
            else:
                raise ValueError('json文件夹中的json文件与img文件夹中的jpg图片的数量/名称不匹配，请核查！')
        else:
            raise FileNotFoundError('请提供json格式的文件，并确保与img文件的数量和名称保持一致，以便数据集处理')
    else:
        label_names = sorted([i[:-4] for i in os.listdir(label_path)]) #将标签和原始图片名称进行排序
        image_names = sorted([i[:-4] for i in os.listdir(img_path)])
        if augment:#需要事先判断是否增强，再来看
            img_files  = sorted(os.listdir(img_path))
            if label_names == image_names:
                assert (len(img_files) >= 4), '训练图片数量不得少于4张！'
                if "_conv" in (img_files[1] and img_files[2] and img_files[3]): #如第2/3/4张图片含增强名称，说明已增强完成
                    pass
                else: #否则进行数据增强。此时是说明完全没有增强处理过，第一次增强。增强中断的情况，则不必进行重命名
                    rename(img_path) #json文件不进行改名。因数据增强仅针对png和jpg图片处理。
                    rename(label_path)
                    print('开始进行数据增强，请耐心等待一段时间，增强完成将会提示...')
                    img_augment(img_path, label_path)
            else: #如label和img数量不一致，要么有错，要么增强中断。
                if "_conv" in img_files[2]: #如果原始图片中的第二张图片中含有增强名称的图片，说明增强中断
                    for img_file in img_files: #处理原始图片
                        if "_conv" in img_file: #只要增强的名称存在图片中，就说明图片已增强过
                            os.remove(img_path + img_file)
                    for label_file in os.listdir(label_path): #处理标签图片
                        if "_conv" in label_file:
                            os.remove(label_path + label_file)
                    # 文件删除干净后，再次进行判断是否名称有误
                    label_names = sorted([i[:-4] for i in os.listdir(label_path)])
                    image_names = sorted([i[:-4] for i in os.listdir(img_path)])
                    if label_names == image_names:
                        print('开始进行数据增强, 请耐心等待一段时间，增强完成将会提示...')
                        img_augment(img_path, label_path)
                    else: #这里的判断是基于删除增强文件后的判断图片数量是否一致，属于极端情况。
                        raise ValueError('标签文件夹中的png图片与img文件夹中的jpg图片的数量/名称不匹配，请核查！')
                else:
                    raise ValueError('标签文件夹中的png图片与img文件夹中的jpg图片的数量/名称不匹配，请核查！')
            print('数据已增强完毕！')
        else: #如不需要数据增强，则直接判断png图片是否与jpg图片一致。不需要数据增强的话，图片不必重命名处理
            if label_names != image_names:
                raise ValueError('label文件夹中的png图片数量与img文件夹中的jpg图片数量/名称不匹配，请核查！')

def label_colormap(N=256): #类别设置颜色
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)
    cmap = np.zeros((N, 3))
    for i in range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap    

def label2rgb(lbl, img = None, n_labels = None, alpha=0.4, colormap=None, mask_save = False): #alpha代表新图中，label的占比
    if n_labels is None:
        n_labels = len(np.unique(lbl))
    colormap = (colormap * 255).astype(np.uint8)
    lbl_viz = colormap[lbl]
    lbl_viz[lbl == -1] = (0, 0, 0)  # unlabeled
    if not mask_save:
        lbl_viz = alpha * lbl_viz + (1 - alpha) * img
    lbl_viz = lbl_viz.astype(np.uint8)
    return lbl_viz

def draw_label(label, img, label_names, colormap):
    backend_org = plt.rcParams['backend']
    plt.switch_backend('agg')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0,wspace=0, hspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    label_viz = label2rgb(label, img, len(label_names), colormap=colormap)
    plt.imshow(label_viz)
    plt.axis('off')
    plt_handlers = []
    plt_titles = []
    for label_value, label_name in enumerate(label_names):
        if label_value not in label:
            continue
        fc = colormap[label_value]
        p = plt.Rectangle((0, 0), 1, 1, fc=fc)
        plt_handlers.append(p)
        plt_titles.append('{value}: {name}'
                        .format(value=label_value, name=label_name))
    plt.legend(plt_handlers, plt_titles, loc='lower right', framealpha=.5)
    f = io.BytesIO()
    plt.savefig(f, bbox_inches='tight', pad_inches=0)
    plt.cla()
    plt.close()
    plt.switch_backend(backend_org)
    out_size = (label_viz.shape[1], label_viz.shape[0]) #就是原图尺寸
    out = Image.open(f).resize(out_size, Image.BILINEAR).convert('RGB')
    out = np.asarray(out)
    return out

def dense_crf(probs, img=None, n_iters = 10, n_classes = 5, #对预测结果进行CRF后处理
                sxy_gaussian=(2, 2), 
                compat_gaussian = 2,
                kernel_gaussian=dcrf.DIAG_KERNEL,
                normalisation_gaussian=dcrf.NORMALIZE_SYMMETRIC,
                sxy_bilateral = (10, 10), 
                compat_bilateral = 10,
                srgb_bilateral = (50, 50, 50),
                kernel_bilateral = dcrf.DIAG_KERNEL,
                normalisation_bilateral = dcrf.NORMALIZE_SYMMETRIC):
    _, h, w, _ = probs.shape
    probs = probs[0].transpose(2, 0, 1).copy(order='C') # Need a contiguous array. 这里将预测结果进行降维，并将通道调至前面
    
    d = dcrf.DenseCRF2D(w, h, n_classes) # Define DenseCRF model.
    U = -np.log(probs) # Unary potential.
    U = U.reshape((n_classes, -1)) # Needs to be flat.
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
                        kernel=kernel_gaussian, normalization=normalisation_gaussian)
    if img is not None:
        assert(img.shape[1:3] == (h, w)), "The image height and width must coincide with dimensions of the logits."
        d.addPairwiseBilateral(sxy=sxy_bilateral, compat=compat_bilateral,
                            kernel=kernel_bilateral, normalization=normalisation_bilateral,
                            srgb=srgb_bilateral, rgbim=img[0])
    Q = d.inference(n_iters)
    preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w)).transpose(1, 2, 0)
    return np.expand_dims(preds, 0)