# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license
import torch
from .yolo_interface import YoloInterface
from ultralytics.utils import ops as uops
from boxmot.utils import ops
from ultralytics.engine.results import Results
import numpy as np

class Yolov5Strategy(YoloInterface):
    
    pt = False
    stride = 32
    fp16 = False
    triton = False

    names = {0: 'ship_misc_misc', 1: 'cargo_ship_misc', 2: 'bulk_carrier', 3: 'fishing_ship', 4: 'other_structure', 5: 'lighthouse', 6: 'ferry', 7: 'wharf', 8: 'yacht', 9: 'boat', 10: 'combat_support_ship', 11: 'buoy', 12: 'destroyer', 13: 'warship_misc', 14: 'beacon', 15: 'light_pole', 16: 'other_float', 17: 'leisure_ship_misc', 18: 'scout_boat', 19: 'light_beacon'}


    def __init__(self, model, device, args):
        self.args = args
        self.pt = False
        self.stride = 32  # max stride in YOLOX

        # self.model = torch.hub.load('ultralytics/yolov5', 'custom', r"yolov5n.pt", force_reload=True)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', model, force_reload=True)
        # self.model = torch.hub.load('ultralytics/yolov5', 'custom', model, force_reload=False)
        # self.model = torch.hub.load('C:/Users/User/Desktop/24-1/해상AI/code/ocean_ai/yolo_tracking-10.0.46/examples/detectors',
        #                             'custom', model, source='local')
        self.model.eval()

        self.model.conf = args.conf
        self.model.iou = args.iou
        self.model.classes = args.classes


    @torch.no_grad()
    def __call__(self, im, augment, visualize):
        # im = im.half() if self.fp16 else im.float()  # uint8 to fp16/32
        # im /= 255  # 0 - 255 to 0.0 - 1.0
        # if len(im.shape) == 3:
        #     im = im[None]  # expand for batch dim

        preds = self.model(im)
        # np.set_printoptions(threshold=np.inf, linewidth=np.inf) 
        # print("################디버깅preds ##########")
        # print(preds.cpu().detach().numpy())

        return preds

    def postprocess(self, path, preds, im, im0s):

        preds = ops.non_max_suppression(preds, self.args.conf, self.args.iou, self.args.classes, self.args.agnostic_nms)
        results = []
        for i, pred in enumerate(preds):

            if pred is None:
                pred = torch.empty((0, 6))
                r = Results(
                    path=path,
                    boxes=pred,
                    orig_img=im0s[i],
                    names=self.names
                )
                results.append(r)
            else:
                # (x, y, x, y, conf, obj, cls) --> (x, y, x, y, conf, cls)
                # print(pred)
                # print(np.shape(pred))
                # pred[:, 4] = pred[:, 4] * pred[:, 5]
                # pred = pred[:, [0, 1, 2, 3, 4, 6]]

                pred[:, :4] = uops.scale_boxes(im.shape[2:], pred[:, :4], im0s[i].shape).round()

                r = Results(
                    path=path,
                    boxes=pred,
                    orig_img=im0s[i],
                    names=self.names
                )
            results.append(r)
        return results

    

    def warmup(self, imgsz):
        pass
