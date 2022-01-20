模型的训练和检测，均需根据自身情况修改basicSet.json文件，来分别设定训练文件的存放位置、图片大小，类别名称，使用的主干网络名称，
batch-size, epochs等。然后运行代码，可实现一键完成json文件转换、数据集的增强，模型训练操作。
主干网络名称和模型名称，init_with均对大小写、空格不敏感，方便用户的输入使用。
basicSet.json文件的设置分成三部分，中间断行隔开，方便设置。第一部分为训练和检测需要用到的公共设置部分；
第二部分为训练模型需要用的设置部分；第三部分为检测需要用到的设置部分；

1） 公共部分参数设置说明：
    A. classes为类别名称设置，其中__background__为背景名称，不得更改。后面的名称则根据自身的实际情况设定。
       应确保填写完整的类别名称，并与实际标注时的名称保持一致，不应缺漏，否则json文件转换时会报错。

    B. img_height和img_width为初始设定的图片大小。训练和检测图片允许大小不统一，但该数值设置后，应确保训练和检测设置的数值一致；
       代码会根据该设定值，将输入的图片统一规整化成设定大小，再纳入模型训练和检测。检测输出仍旧是原始图片大小，而非设定大小。
       但不建议训练与检测图片尺寸有差别，会影响实际检测效果。

    C. 主干网络bone_name的名称可选参数为：
       ["new_modelv1", "efficientnet_b0", "efficientnet_b2", "mobilenetv2", "mobilenetv3", "inceptionresnetv2"], 共6种。
       模型名称model_name的可选参数为：
       ["deeplab", "unet", "pspnet"]; 
       其中new_modelv1为自行设计的网络，这样可以组成18种模型进行训练。可根据自身的情况，来分别设定，并查看对比不同模型效果。

    D. 文件路径设置需注意，需为双斜杠或单反斜杠；
       训练结果的模型文件，会根据log_path的设置，自动在指定文件夹中保存文件，如果log_path为空，则在根目录下自动创建log文件夹。

2) 模型训练部分的参数设置说明：
    A. json_path为labelme标注产生的json文件路径，需要确保json文件的名称与原始图片名称保持一致；
       注意json文件必须用labelme进行标注，才能正常转换，否则出错。
       
    B. train_img_path为原始训练图片存放位置，允许为jpg或png格式。

    C. label_path为标签图片存放位置。如已存在json文件，则设置好该值后，代码会自动创建该文件夹，并将转换出来的png图片存入。
       如已转换生成标签图片，则直接将标签图片存放在该文件夹中即可，代码自动忽略json文件转换。标签图片统一为png格式。

    D. weights_path为预训练权重文件位置。如需要加载该预训练文件，可直接进入训练代码中，解除相应的代码注释，
       则模型会自动加载预训练权重文件（此时，应确保weights_path的预训练权重文件路径和文件名正确）；
       解除注释的代码语句为：model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    E. batch_size和epochs为人为设定的批次文件大小，及训练的轮数。该设定的批次文件大小根据自身的电脑配置决定。
       而训练轮数越高，模型效果越趋近上限。

    F. init_with参数，用来设置模型用于初始训练，还是继续之前的训练。可选参数为：["first", "last"]；

3) 模型检测部分的参数设置说明：
    A. bottom_point和top_point用于设定谷歌地图爬图软件下载时，指定的地图左下角和右上角的所框选区域。如不需要可设置为空值或忽略；

    B. predict_img_path为检测的图片存放位置。如文件夹中已有图片，则代码自动进行检测，不会下载地图。
       但是需注意下，如检测时使用的图片为png格式，需进入检测部分的代码，修改如下语句：---将"/*.jpg"更换成"/*.png".
       imgs = glob.glob(roots + "/*.jpg") 

    C. result_path为检测结果文件保存位置，设定后，代码会自动创建该文件夹。

    D. area_per_pixel为检测图片中每个像素点代表的面积大小。如有需求，可根据自身情况进行设定。
       模型会自动根据设定的类别，分类统计各类别的面积，及总面积大小。

4）其他补充说明：
    A. 训练代码默认进行数据增强处理，数据增强将会耗时一段时间，增强后的图片会保存在原始图片及标签图片的相应文件夹中。
       训练数据集至少要有4张图片，否则会报错，主要是为了确保能正常识别数据增强。
       代码可以实现15倍数据量的增强，其中包含了翻转、增加高斯噪声、模糊、同态滤波、通道互换等。
       原始图片的文件名称如含有下划线，则会自动替换成横杠线，会尽量保持原有文件名称。
       如增强过程中，代码中断，下次重新运行代码，会自动处理并继续增强。
       如不需要进行数据增强，可进入train的代码，将如下代码中的augment设置为False
       image_treat(img_path, label_path, augment = True) #图片转换及增强处理

    B. 检测时，如需指定保存的结果，可进入代码，对该语句进行设置：
       predict(model_name, mask_save = False, combine_save = True, crf_treat = True, keep_path_struct = True)
       mask_save代表保存单独的检测结果与否，combine_save代表保存原始和检测的融合结果与否, crf_treat代表是否进行CRF处理,
       keep_path_struct代表检测结果是否按照检测文件的存放结构来保存检测结果，False则统一保存到result_path文件夹下。
       一般建议进行CRF处理，可以使得检测结果边缘平滑，减少误判，但该CRF需要手动调参---调节参数为sxy_gaussian，compat_gaussian等
       单独保存的检测结果像素值根据类别数量被放大，以便肉眼观察。融合结果为jpg格式，检测结果为png格式图片。
       保存的mask图片为三通道彩图，如不需三通道彩图，可进入代码，将下面语句注释掉：
       mask = label2rgb(pr, n_labels = len(classes), colormap = colors, mask_save = mask_save)
       检测时，不论指定的文件夹下为图片，还是多级子目录的图片存放，均可自动进行检测，检测结果也会以相同的结构保存在检测文件夹下。
       同时保证文件名称与原始名称一致，方便核对。

    C. 训练或检测出错的话，代码会在当前目录下，自动生成报错的json文件。可打开查看出错原因。
       问题解决后，再次运行代码，则会自动删除该报错文件。
