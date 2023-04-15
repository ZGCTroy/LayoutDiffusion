import gradio as gr
import numpy as np
import cv2
from layout_diffusion.util import fix_seed
import pandas as pd


def get_demo(layout_to_image_generation_fn, cfg, model_fn, noise_schedule):
    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
                   'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                   'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                   'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                   'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                   'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet', 'cage', 'cardboard',
                   'carpet', 'ceiling-other', 'ceiling-tile', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain', 'desk-stuff', 'dirt',
                   'door-stuff', 'fence', 'floor-marble', 'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower', 'fog', 'food-other', 'fruit',
                   'furniture-other', 'grass', 'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal', 'mirror-stuff', 'moss',
                   'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow', 'plant-other', 'plastic', 'platform', 'playingfield', 'railing',
                   'railroad', 'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper', 'snow',
                   'solid-other', 'stairs', 'stone', 'straw', 'structural-other', 'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable',
                   'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
                   'window-blind', 'window-other', 'wood', 'other', 'image', 'pad']

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 0, 0), (0, 128, 0),
              (0, 0, 128), (128, 128, 0), (0, 128, 128), (128, 0, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64)]

    def calculate_bbox(mask, class_name, custom_layout_dict):
        '''
        :param img: RGB
        :param mask: Gray
        :param class_name: str
        :return:
        '''
        custom_layout_dict['num_obj'] += 1

        if np.all(mask == 0):
            custom_layout_dict['obj_class'].append('pad')
            custom_layout_dict['obj_bbox'].append([0.0, 0.0, 0.0, 0.0])
            return custom_layout_dict

        # 对灰度图像应用阈值处理，创建 mask
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        all_unique_contours = np.vstack(contours)
        all_unique_contours = np.unique(all_unique_contours, axis=0)
        x, y, w, h = cv2.boundingRect(all_unique_contours)

        custom_layout_dict['obj_class'].append(class_name)
        custom_layout_dict['obj_bbox'].append([float(_) / 256.0 for _ in [x, y, x + w, y + h]])

        return custom_layout_dict

    def update_layout_image(custom_layout_dataframe):
        '''
        :param img: RGB
        :param custom_layout_dataframe: dict
        :return:
        '''
        init_layout_image = np.ones((256, 256, 3), dtype=np.uint8) * 255

        bgr_img = cv2.cvtColor(init_layout_image, cv2.COLOR_RGB2BGR)

        print(custom_layout_dataframe)
        for obj_id in range(len(custom_layout_dataframe)):
            obj_class = custom_layout_dataframe['obj_class'][obj_id]
            if obj_class == 'pad':
                continue

            color = colors[obj_id]
            x0, y0, x1, y1 = custom_layout_dataframe['obj_bbox_x0'][obj_id], custom_layout_dataframe['obj_bbox_y0'][obj_id], \
                    custom_layout_dataframe['obj_bbox_x1'][obj_id], custom_layout_dataframe['obj_bbox_y1'][obj_id]

            x0, y0, x1, y1 = int(float(x0) * 256), int(float(y0) * 256), int(float(x1) * 256), int(float(y1) * 256)

            x, y, w, h = x0, y0, x1 - x0, y1 - y0

            cv2.rectangle(bgr_img, (x, y), (x + w, y + h), color, 2)
            text = f"{obj_class}"
            text_w, text_h = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(bgr_img, (x + w - text_w, y + h - 2 * text_h), (x + w, y + h), color, -1)
            cv2.putText(bgr_img, text, (x + w - text_w, y + h - text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        return rgb_img

    with gr.Blocks() as demo:
        gr.Markdown(
            """
        # LayoutDiffusion
        Get "layout image" and then "layout-to-image generation".
        Please note that the model is not pretrained on models like Stable Diffusion and is trained from scratch only on a small subset of COCO-stuff dataset in pixel-space. 
        Therefore, fine-tuning on Stable Diffusion, training on a larger dataset, or being guided with text prompt can definitely result in better performance.
        """
        )
        num_obj = 8
        with gr.Row():
            with gr.Column():
                custom_layout_dataframe = gr.Dataframe(
                    headers=["obj_id", "obj_class", "obj_bbox_x0","obj_bbox_y0","obj_bbox_x1","obj_bbox_y1"],
                    datatype=["number", "str", "number","number","number","number"],
                    row_count=(num_obj + 2, "fixed"),
                    col_count=(6, "fixed"),
                )

            with gr.Column():
                with gr.Row():
                    layout_image = gr.Image(label='Layout Image', shape=(256, 256), value=np.ones((256, 256, 3), dtype=np.uint8) * 255).style(width=256, height=256)
                    generated_image = gr.Image(label='Generated Image', shape=(256, 256)).style(width=256, height=256)
                with gr.Row():
                    button_to_update_layout_image = gr.Button(value='Update Layout Image')
                    button_to_update_layout_image.click(fn=update_layout_image, inputs=[custom_layout_dataframe], outputs=[layout_image])
                    generate_button = gr.Button(value='Generate using Layout Image')
                with gr.Row():
                    classifier_free_scale = gr.Slider(value=1.0, minimum=0.5, maximum=2.0, step=0.1, label='Classifier free scale')
                    steps = gr.Slider(value=25, minimum=25, maximum=200, label='Steps')

                with gr.Row():
                    seed = gr.Number(value=2333, precision=0, label='Seed')
                    button_to_random_seed = gr.Button(value='random seed')
                    button_to_random_seed.click(
                        fn=lambda : int(np.random.randint(low=1, high=9999999, size=1)),
                        outputs=[seed],
                    )

        gr.Examples(
            examples=[
                [
                    [[0, 'image', 0.0, 0.0, 1.0, 1.0],
                     [1, 'tree', 0.0, 0.0, 1.0, 0.2750000059604645],
                     [2, 'grass', 0.0, 0.14166666567325592, 1.0, 1.0, ],
                     [3, 'zebra', 0.18989062309265137, 0.23635415732860565, 0.8281093835830688, 0.8176667094230652],
                     [4, 'pad', 0.0, 0.0, 0.0, 0.0],
                     [5, 'pad', 0.0, 0.0, 0.0, 0.0],
                     [6, 'pad', 0.0, 0.0, 0.0, 0.0],
                     [7, 'pad', 0.0, 0.0, 0.0, 0.0],
                     [8, 'pad', 0.0, 0.0, 0.0, 0.0],
                     [9, 'pad', 0.0, 0.0, 0.0, 0.0]],
                    1.0, 25, 3739045
                ],
                [
                    [[0, 'image', 0.0, 0.0, 1.0, 1.0],
                     [1, 'train', 0.09859374910593033, 0.24045585095882416, 0.9391249418258667, 0.8741595149040222],
                     [2, 'pavement', 0.0, 0.49857550859451294, 1.0, 1.0],
                     [3, 'sky-other', 0.0, 0.0, 0.515625, 0.5441595315933228],
                     [4, 'tree', 0.28125, 0.0, 1.0, 0.41310539841651917],
                     [5, 'wall-other', 0.8999999761581421, 0.3817663788795471, 1.0, 0.692307710647583],
                     [6, 'pad', 0.0, 0.0, 0.0, 0.0],
                     [7, 'pad', 0.0, 0.0, 0.0, 0.0],
                     [8, 'pad', 0.0, 0.0, 0.0, 0.0],
                     [9, 'pad', 0.0, 0.0, 0.0, 0.0]],
                    1.0, 25, 7296572
                ],
                [
                    [[0, 'image', 0.0, 0.0, 1.0, 1.0],
                     [1, 'desk-stuff', 0.0, 0.5791666507720947, 1.0, 1.0],
                     [2, 'laptop', 0.001687500043772161, 0.41574999690055847, 0.3943749964237213, 0.9820416569709778],
                     [3, 'window-blind', 0.0, 0.0, 0.25312501192092896, 0.4791666567325592],
                     [4, 'plastic', 0.20000000298023224, 0.05000000074505806, 1.0, 0.7208333611488342],
                     [5, 'wall-concrete', 0.17812499403953552, 0.0, 1.0, 0.5916666388511658],
                     [6, 'keyboard', 0.503531277179718, 0.6351667046546936, 0.9857500195503235, 0.824833333492279],
                     [7, 'tv', 0.39268749952316284, 0.04774999991059303, 0.7415624856948853, 0.4207916557788849],
                     [8, 'light', 0.840624988079071, 0.02083333395421505, 0.925000011920929, 0.42500001192092896],
                     [9, 'pad', 0.0, 0.0, 0.0, 0.0]],
                    1.0, 25, 290
                ],
            ],
            inputs=[
                custom_layout_dataframe, classifier_free_scale, steps, seed
            ],
            label='Update Layout Using Examples'
        )

        obj_class_list, obj_mask_list, obj_bbox_list = [], [], []
        with gr.Accordion('Construct a Layout by drawing object one-by-one', visible=True, open=False):
            gr.Markdown(
                """
            Draw each object and click "Update Layout Image by Painting". It is recommended to draw multiple objects that cover the whole image.
            The stuff categories are necessary to act as the background.
            """
            )
            button_to_construct_layout_by_drawing = gr.Button(value='Update Layout Image by Painting')

            with gr.Row():
                for obj_id in range(num_obj):
                    with gr.Column(variant='panel'):
                        obj_mask_list.append(gr.Image(source='canvas', tool='color-sketch', type='numpy', invert_colors=True, shape=(256, 256), label='obj_id={}'.format(obj_id + 1)))
                        obj_class_list.append(gr.Dropdown(choices=class_names, value='pad', label='obj_class'))

        def custom_layout_dict_to_dataframe(custom_layout_dict):
            custom_layout_dataframe = {
                'obj_id': [],
                'obj_class': [],
                'obj_bbox_x0': [],
                'obj_bbox_y0': [],
                'obj_bbox_x1': [],
                'obj_bbox_y1': [],
            }
            custom_layout_dataframe['obj_id'].append(0)
            custom_layout_dataframe['obj_class'].append('image')
            custom_layout_dataframe['obj_bbox_x0'].append(0)
            custom_layout_dataframe['obj_bbox_y0'].append(0)
            custom_layout_dataframe['obj_bbox_x1'].append(1)
            custom_layout_dataframe['obj_bbox_y1'].append(1)

            for obj_id in range(custom_layout_dict['num_obj']):
                custom_layout_dataframe['obj_id'].append(obj_id + 1)
                custom_layout_dataframe['obj_class'].append(custom_layout_dict['obj_class'][obj_id])
                custom_layout_dataframe['obj_bbox_x0'].append(custom_layout_dict['obj_bbox'][obj_id][0])
                custom_layout_dataframe['obj_bbox_y0'].append(custom_layout_dict['obj_bbox'][obj_id][1])
                custom_layout_dataframe['obj_bbox_x1'].append(custom_layout_dict['obj_bbox'][obj_id][2])
                custom_layout_dataframe['obj_bbox_y1'].append(custom_layout_dict['obj_bbox'][obj_id][3])

            custom_layout_dataframe['obj_id'].append(custom_layout_dict['num_obj'] + 1)
            custom_layout_dataframe['obj_class'].append('pad')
            custom_layout_dataframe['obj_bbox_x0'].append(0)
            custom_layout_dataframe['obj_bbox_y0'].append(0)
            custom_layout_dataframe['obj_bbox_x1'].append(0)
            custom_layout_dataframe['obj_bbox_y1'].append(0)

            return pd.DataFrame(custom_layout_dataframe)

        def custom_layout_dataframe_to_dict(custom_layout_dataframe):
            obj_bbox = []
            for x0,y0,x1,y1 in zip(custom_layout_dataframe['obj_bbox_x0'],custom_layout_dataframe['obj_bbox_y0']
                    ,custom_layout_dataframe['obj_bbox_x1'], custom_layout_dataframe['obj_bbox_y1']):
                obj_bbox.append([float(x0),float(y0),float(x1),float(y1)])
            custom_layout_dict = {
                'num_obj': len(custom_layout_dataframe),
                'obj_class': custom_layout_dataframe['obj_class'],
                'obj_bbox': obj_bbox
            }
            return custom_layout_dict

        def update_layout_image_by_drawing(*args, **kwargs):
            custom_layout_dict = {
                'num_obj': 0,
                'obj_class': [],
                'obj_bbox': []
            }

            for obj_id in range(num_obj):
                obj_class = args[obj_id]
                obj_mask = args[num_obj + obj_id]

                if obj_mask is None:
                    obj_mask = np.zeros((256, 256, 3), dtype=np.uint8)

                custom_layout_dict = calculate_bbox(mask=obj_mask, class_name=obj_class, custom_layout_dict=custom_layout_dict)

            print(custom_layout_dict)

            custom_layout_dataframe = custom_layout_dict_to_dataframe(custom_layout_dict)
            updated_layout_image = update_layout_image(custom_layout_dataframe)

            return updated_layout_image, custom_layout_dataframe

        button_to_construct_layout_by_drawing.click(fn=update_layout_image_by_drawing, inputs=obj_class_list + obj_mask_list, outputs=[layout_image, custom_layout_dataframe])

        def layout_to_image_generation(custom_layout_dataframe, classifier_free_scale, steps, seed):
            if cfg.sample.fix_seed:
                fix_seed(seed)

            cfg.sample.classifier_free_scale = classifier_free_scale
            cfg.sample.timestep_respacing[0] = str(steps)
            cfg.sample.sample_method = 'dpm_solver'

            custom_layout_dict = custom_layout_dataframe_to_dict(custom_layout_dataframe)

            return layout_to_image_generation_fn(cfg, model_fn, noise_schedule, custom_layout_dict)

        generate_button.click(
            fn=layout_to_image_generation, inputs=[custom_layout_dataframe, classifier_free_scale, steps, seed], outputs=[generated_image]
        )

    return demo
