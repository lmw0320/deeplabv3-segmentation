import sys, os
path = os.path.dirname(__file__)
sys.path.append(path) #临时加入环境变量，以便下面代码导入模块
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

#再次训练前，事先删除报错文件。
if os.path.exists(r'%s\train_error.json'%path):
    os.remove(r'%s\train_error.json'%path)

try:
    import cv2, json, time, traceback, copy, shutil, glob
    import numpy as np
    import pandas as pd
    from keras import backend as K
    from keras.optimizers import Adam
    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping, Callback

    from nets.utils.data_treat import image_treat, get_data, letterbox_image
    from nets.utils.function import focal_loss, find_last
    from deeplabv3 import Deeplabv3
    from unet import Unet
    from pspnet import PSPnet

    from warnings import simplefilter
    simplefilter(action = 'ignore', category = FutureWarning)

except (ModuleNotFoundError, ImportError) as reason:
    print(str(reason))
    print('扩展包未安装或安装不正确，请重新安装相应的扩展包！')
    with open(r'%s\train_error.json'%path, 'w') as f:
        f.write(traceback.format_exc())
        f.write('\n')
        f.write('错误发生时间：%s'%time.strftime('%Y-%m-%d %H:%M:%S'))
    sys.exit()

try:
    def get_info():
        filepath = json.load(open(r"%s\basicSet.json"%path, encoding= 'utf-8'))[0] #读取路径配置文件中的内容
        weights_path = filepath['weights_path']  #模型预训练权重文件
        json_path = filepath['json_path'] + "\\" #labelme标注后的json文件保存位置
        img_path = filepath['train_img_path'] + "\\" #训练图片保存位置
        label_path = filepath['label_path'] + "\\" #标签图片保存位置
        log_path =  filepath['log_path'] #模型文件位置, 这里先不增加反斜杠的后缀，方便后面判断是否设置log_path
        classes = filepath['classes'] #模型的类别名称，名称必须与标注时定义的名称保持一致
        HEIGHT = filepath['img_height'] #指定模型训练图片的高度
        WIDTH = filepath['img_width'] #指定模型训练图片的宽度
        batch_size = filepath['batch_size'] #数据集的切分块大小
        epochs = filepath['epochs'] #训练模型的轮数
        init_with = filepath['init_with'] #设置模型开始训练时的状态
        bone_name = filepath['bone_name'] #主干网络名称
        model_name = filepath['model_name'] #模型名称

        model_name = model_name.lower().replace(" ", "") #忽略大小写的敏感，并忽略字符间的空格，方便用户的输入
        bone_name = bone_name.lower().replace(" ", "")
        init_with = init_with.lower().replace(" ", "")
        log_path = log_path.replace(" ", "")

        if model_name not in ["deeplabv3", "unet", "pspnet"]: 
            raise ValueError('请输入正确的模型名称(大小写不敏感)，可选项为：["deeplabv3", "unet", "pspnet"]')
        if bone_name not in ["new_modelv1", "efficientnet_b0", "efficientnet_b2", "mobilenetv2", "mobilenetv3", "inceptionresnetv2"]:
            raise ValueError('请输入正确的主干网络名称(大小写不敏感)，可选项为：\
                            ["new_modelv1", "efficientnet_b0", "efficientnet_b2", "mobilenetv2", "mobilenetv3", "inceptionresnetv2"]')
        if init_with not in ["first", 'last']:
            raise ValueError('请正确设置模型的训练状态(大小写不敏感)，可选项为：["first", "last"]')
        return weights_path, json_path, img_path, label_path, log_path, classes, HEIGHT, WIDTH, batch_size, epochs,\
             init_with, bone_name, model_name

    def generate_arrays_from_file(data, batch_size): #该函数放在主文件中，也是为了避免传入HEIGHT和WIDTH的参数
        i = 0
        while 1:
            X_train = []
            Y_train = []
            for _ in range(batch_size): # 获取一个batch_size大小的数据
                if i==0:
                    np.random.seed(1)
                    np.random.shuffle(data)
                name = data[i][:-4] #获取文件名
                img = cv2.imread(img_path + data[i], cv2.IMREAD_UNCHANGED)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = letterbox_image(img, (HEIGHT, WIDTH))
                img = img /255 #归一化

                label = cv2.imread(label_path + name + '.png')#方便不同通道数的标签文件，统一读取为三通道矩阵
                label = letterbox_image(label, (HEIGHT, WIDTH))

                label = cv2.resize(label, (HEIGHT, WIDTH))
                seg_labels = np.zeros((HEIGHT, WIDTH, NCLASSES))
                for c in range(NCLASSES):#将每个类别分别装填入对应的类别通道中,各通道类别对应像素值均更改为1
                    seg_labels[: , : , c ] = (label[:, :, 0] == c ).astype(int) 
                seg_labels = np.reshape(seg_labels, (-1, NCLASSES))
                X_train.append(img)
                Y_train.append(seg_labels)

                i = (i+1) % len(data) #读完一个周期后重新开始
            yield (np.array(X_train), np.array(Y_train))

    # Custom loss function, 该损失函数暂时不用，而改用function中的focal_loss.
    #同时，该loss没有整合到function中，是因为这里用到了HEIGHT和WIDTH，无法传参。需放在主文件中，以便获取全局变量。
    def cross_loss(y_true, y_pred):
        crossloss = K.binary_crossentropy(y_true, y_pred)
        loss = K.sum(crossloss)/HEIGHT/WIDTH
        return loss

    def train_model(train_data, val_data, num_train, num_val, epochs, callback = True): #callback用于是否记录训练。
        if model_name == "deeplabv3":
            model = Deeplabv3((HEIGHT, WIDTH, 3), NCLASSES, bone_name)
        elif model_name == 'unet':
            model = Unet((HEIGHT, WIDTH, 3), NCLASSES, bone_name)
        elif model_name == 'pspnet':
            model = PSPnet((HEIGHT, WIDTH, 3), NCLASSES, bone_name)

        epoch = 0
        if init_with == 'first': #init_with selection = ['first', 'last'] 选择模型的训练模式
            print('-'*100)
            # model.load_weights(weights_path, by_name=True, skip_mismatch=True) #加载预训练模型
            print('开始从头训练模型...')
        else:
            model.load_weights(find_last(log_path), by_name=True)
            epoch = int(os.path.basename(find_last(log_path))[:-3].split('_')[-1])
            epochs = epoch + epochs #这样可以确保重新训练时，设置的epochs为实际训练的轮数。
            print('-'*100)
            print('成功加载最新模型, 重新从第%s轮开始训练...'%epoch)

        #保存训练过程
        tbCallBack = TensorBoard(log_dir=log_path + "records/",
                                histogram_freq=0,
                                write_graph=True,
                                write_images=True)
    
        # 保存的方式，1次epoch保存一次
        checkpoint_path = os.path.join(log_path, '%s_%s_*epoch*.h5'%(model_name, bone_name))
        checkpoint_path = checkpoint_path.replace("*epoch*", "{epoch:04d}")
        checkpoint_period = ModelCheckpoint(checkpoint_path,
                                            monitor='val_loss',
                                            save_weights_only=True,
                                            save_best_only=True,
                                            period=1)

        # 学习率下降的方式，val_loss五次不下降就下降学习率继续训练
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      factor=0.5, 
                                      patience=5, 
                                      verbose=1)

        # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=15,
                                       verbose=1)

        callbacks = [checkpoint_period, reduce_lr, early_stopping]
        if callback:
            tbCallBack.set_model(model)
            callbacks.append(tbCallBack)

        print('训练的样本量为：{} 张图片, 验证集的样本量为：{} 张图片，'.format(num_train, num_val))
        print('每份数据集大小为：{} 张图片, 图片大小为: {}'.format(batch_size, (HEIGHT, WIDTH)))
        print('以%s为主干的%s模型正式开始训练，请耐心等待，并注意提示...'%(bone_name, model_name)) #开始训练
        print('-' * 100)

        model.compile(loss = focal_loss,
                optimizer = Adam(lr=1e-3),
                metrics = ['accuracy'])
        model.fit_generator(generate_arrays_from_file(train_data, batch_size),
                steps_per_epoch = max(1, num_train//batch_size),
                validation_data = generate_arrays_from_file(val_data, batch_size),
                validation_steps = max(1, num_val//batch_size),
                epochs = epochs,
                initial_epoch = epoch,
                shuffle = True,
                callbacks = callbacks)

    if __name__ == "__main__":
        t_start = time.time()
        print('代码开始执行，开始时间为: %s'%time.strftime('%Y-%m-%d %H:%M:%S'))
        weights_path, json_path, img_path, label_path, log_path, classes, HEIGHT, WIDTH, batch_size, epochs, \
        init_with, bone_name, model_name = get_info()
        NCLASSES = len(classes)
        if not len(log_path):
            log_path = path + "/%s_%s_logs/"%(model_name, bone_name)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        image_treat(img_path, label_path, json_path, classes, augment = True) #图片转换及增强处理
        train_data, val_data, num_train, num_val = get_data(img_path) #获取训练集和验证集
        train_model(train_data, val_data, num_train, num_val, epochs, callback= False)
        t_end = time.time()
        time_total = pd.to_timedelta(t_end - t_start,unit = 's')
        print('模型训练完毕，结束时间为: %s, 总耗时: %s'%(time.strftime('%Y-%m-%d %H:%M:%S'), time_total.round("s")))

except Exception as reason:
    #系统退出，程序运行人为中断，生成器等特殊问题，不考虑记录。
    with open(r'%s\train_error.json'%path, 'w') as f:
        f.write(traceback.format_exc())
        f.write('\n')
        f.write('错误发生时间：%s'%time.strftime('%Y-%m-%d %H:%M:%S'))
    print('发生错误，原因是 %s'%str(reason))