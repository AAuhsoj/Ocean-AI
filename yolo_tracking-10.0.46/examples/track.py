# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import subprocess

# pip uninstall -y ultralytics 실행
# subprocess.run(["conda", "init"])
# subprocess.run(["conda", "activate", "ocean_ai"])
subprocess.run(["pip", "uninstall", "-y", "ultralytics"])


import argparse
from functools import partial
from pathlib import Path

import torch

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS
from boxmot.utils.checks import TestRequirements
from examples.detectors import get_yolo_inferer

__tr = TestRequirements()
# 여기서 ultralytics 다운
__tr.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install


import shutil

# predictor.py 이동
source_file = 'C:/Users/User/Desktop/24-1/해상AI/code/ocean_ai/predictor_origin_debug.py'
destination_dir = 'C:/Users/User/anaconda3/envs/ocean_ai/Lib/site-packages/ultralytics/engine/predictor.py'
shutil.copy(source_file, destination_dir)

# plotting.py 이동
source_file = 'C:/Users/User/Desktop/24-1/해상AI/code/ocean_ai/plotting_origin_debug.py'
destination_dir = 'C:/Users/User/anaconda3/envs/ocean_ai/Lib/site-packages/ultralytics/utils/plotting.py'
shutil.copy(source_file, destination_dir)


# 여기서 ultralytics 설치 후 메모리에 올라감
# 아래 라인 실행 전에 메모리에 올라갈 파일을 변경해줘야함.
from ultralytics import YOLO
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box

from examples.utils import write_mot_results


global img_check_debug
img_check_debug = False


def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = \
        ROOT /\
        'boxmot' /\
        'configs' /\
        (predictor.custom_args.tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers


@torch.no_grad()
def run(args):

    yolo = YOLO(
        args.yolo_model if 'yolov8' in str(args.yolo_model) else 'yolov8n.pt',
    )

    results = yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        show=args.show,
        stream=True,
        device=args.device,
        show_conf=args.show_conf,
        save_txt=args.save_txt,
        show_labels=args.show_labels,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )

    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))

    
    if 'yolov8' not in str(args.yolo_model):
        # replace yolov8 model
        m = get_yolo_inferer(args.yolo_model)
        # ultralytics 업데이트 진행됨
        model = m(
            model=args.yolo_model,
            device=yolo.predictor.device,
            args=yolo.predictor.args
        )
        yolo.predictor.model = model

    # from pytorch_grad_cam import GradCAM
    # from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
    # import matplotlib.pyplot as plt
    # from PIL import Image
    # from torchvision import transforms

    # model = model.model.model if hasattr(model.model, 'model') else model.model
    



    # def apply_gradcam(model, img):
    #     target_layers = [model.model]  # 마지막 레이어 사용, 필요에 따라 변경
    #     cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    #     grayscale_cam = cam(input_tensor=img)[0, :]  # 첫 번째 이미지에 대한 CAM 얻기

    #     img = img.squeeze(0).permute(1, 2, 0).numpy()  # 이미지 텐서를 numpy 배열로 변환
    #     img = (img - img.min()) / (img.max() - img.min())  # Normalize
    #     visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    #     return visualization


    # print(target_layers)

    # store custom args in predictor
    yolo.predictor.custom_args = args

    # 업데이트가 진행돼도
    # 실행되는건 기존에 메모리에 올라가있던 예전 파일이 실행되기 때문에
    # 디렉토리를 원래것으로 교체
    
    # 디렉토리 교체
    
    # 소스 디렉토리 경로
    source_dir = 'C:/Users/User/Desktop/24-1/해상AI/code/ocean_ai/ultralytics_old'

    # 대상 디렉토리 경로
    destination_dir = 'C:/Users/User/anaconda3/envs/ocean_ai/Lib/site-packages/ultralytics'

    # 대상 디렉토리 삭제
    shutil.rmtree(destination_dir, ignore_errors=True)

    # 디렉토리 복사
    shutil.copytree(source_dir, destination_dir)

    # 디렉토리를 교체하고
    # 디버깅을 위해 check용 파일로 교체
    # predictor.py 이동
    source_file = 'C:/Users/User/Desktop/24-1/해상AI/code/ocean_ai/predictor_origin_debug.py'
    destination_dir = 'C:/Users/User/anaconda3/envs/ocean_ai/Lib/site-packages/ultralytics/engine/predictor.py'
    shutil.copy(source_file, destination_dir)

    # plotting.py 이동
    source_file = 'C:/Users/User/Desktop/24-1/해상AI/code/ocean_ai/plotting_origin_debug.py'
    destination_dir = 'C:/Users/User/anaconda3/envs/ocean_ai/Lib/site-packages/ultralytics/utils/plotting.py'
    shutil.copy(source_file, destination_dir)


    global frame_idx
    for frame_idx, r in enumerate(results):
        print("\n")
        if frame_idx == 50:
            img_check_debug = True
        else:
            img_check_debug = False

        if r.boxes.data.shape[1] == 7:

            if yolo.predictor.source_type.webcam or args.source.endswith(VID_FORMATS):
                p = yolo.predictor.save_dir / 'mot' / (args.source + '.txt')
                yolo.predictor.mot_txt_path = p
            elif 'MOT16' or 'MOT17' or 'MOT20' in args.source:
                p = yolo.predictor.save_dir / 'mot' / (Path(args.source).parent.name + '.txt')
                yolo.predictor.mot_txt_path = p

            if args.save_mot:
                write_mot_results(
                    yolo.predictor.mot_txt_path,
                    r,
                    frame_idx,
                )

            if args.save_id_crops:
                for d in r.boxes:
                    print('args.save_id_crops', d.data)
                    save_one_box(
                        d.xyxy,
                        r.orig_img.copy(),
                        file=(
                            yolo.predictor.save_dir / 'crops' /
                            str(int(d.cls.cpu().numpy().item())) /
                            str(int(d.id.cpu().numpy().item())) / f'{frame_idx}.jpg'
                        ),
                        BGR=True
                    )

    if args.save_mot:
        print(f'MOT results saved to {yolo.predictor.mot_txt_path}')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8n',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--save-mot', action='store_true',
                        help='save tracking results in a single txt file')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--vid_stride', default=1, type=int,
                        help='video frame-rate stride')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)
