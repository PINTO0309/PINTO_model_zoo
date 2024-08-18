import torch
from models.yolo import Model
import argparse

def main(args):
    type: str = args.type
    cfg: str = args.cfg
    check_point_file: str = args.weights
    save_pt_file_name = args.save

    ckpt = torch.load(check_point_file, map_location='cpu')
    names = ckpt['model'].names
    nc = ckpt['model'].nc

    device = torch.device("cpu")
    model = Model(cfg, ch=3, nc=nc, anchors=3)
    print('')
    print(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ nc: {nc}')
    print('')
    model = model.to(device)
    _ = model.eval()
    model.names = names
    model.nc = nc

    idx = 0
    if type in ['n', 't', 's']:
        for k, v in model.state_dict().items():
            if "model.{}.".format(idx) in k:
                if idx < 22:
                    kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.cv2.".format(idx) in k:
                    kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+7))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.cv3.".format(idx) in k:
                    kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+7))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.dfl.".format(idx) in k:
                    kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+7))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
            else:
                while True:
                    idx += 1
                    if "model.{}.".format(idx) in k:
                        break
                if idx < 22:
                    kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.cv2.".format(idx) in k:
                    kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+7))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.cv3.".format(idx) in k:
                    kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+7))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.dfl.".format(idx) in k:
                    kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+7))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")

    elif type == 'm':
        for k, v in model.state_dict().items():
            if "model.{}.".format(idx) in k:
                if idx < 22:
                    kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+1))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.cv2.".format(idx) in k:
                    kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+16))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.cv3.".format(idx) in k:
                    kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+16))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.dfl.".format(idx) in k:
                    kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+16))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
            else:
                while True:
                    idx += 1
                    if "model.{}.".format(idx) in k:
                        break
                if idx < 22:
                    kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+1))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.cv2.".format(idx) in k:
                    kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+16))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.cv3.".format(idx) in k:
                    kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+16))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.dfl.".format(idx) in k:
                    kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+16))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")

    elif type == 'c':
        for k, v in model.state_dict().items():
            if "model.{}.".format(idx) in k:
                if idx < 22:
                    kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+1))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                elif "model.{}.cv2.".format(idx) in k:
                    kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+16))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                elif "model.{}.cv3.".format(idx) in k:
                    kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+16))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                elif "model.{}.dfl.".format(idx) in k:
                    kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+16))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            else:
                while True:
                    idx += 1
                    if "model.{}.".format(idx) in k:
                        break
                if idx < 22:
                    kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+1))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                elif "model.{}.cv2.".format(idx) in k:
                    kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+16))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                elif "model.{}.cv3.".format(idx) in k:
                    kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+16))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                elif "model.{}.dfl.".format(idx) in k:
                    kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+16))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]

    elif type == 'e':
        for k, v in model.state_dict().items():
            if "model.{}.".format(idx) in k:
                if idx < 29:
                    kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif idx < 42:
                    kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+7))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.cv2.".format(idx) in k:
                    kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+7))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.cv3.".format(idx) in k:
                    kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+7))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.dfl.".format(idx) in k:
                    kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+7))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
            else:
                while True:
                    idx += 1
                    if "model.{}.".format(idx) in k:
                        break
                if idx < 29:
                    kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif idx < 42:
                    kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+7))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.cv2.".format(idx) in k:
                    kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+7))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.cv3.".format(idx) in k:
                    kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+7))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
                elif "model.{}.dfl.".format(idx) in k:
                    kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+7))
                    model.state_dict()[k] -= model.state_dict()[k]
                    model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                    print(k, "perfectly matched!!")
    _ = model.eval()

    m_ckpt = {'model': model.half(),
            'optimizer': None,
            'best_fitness': None,
            'ema': None,
            'updates': None,
            'opt': None,
            'git': None,
            'date': None,
            'epoch': -1}
    torch.save(m_ckpt, save_pt_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='t', help='convert model type (t or e)')
    parser.add_argument('--cfg', type=str, default='./models/detect/gelan-t.yaml', help='model.yaml path')
    parser.add_argument('--weights', type=str, default='./best-t.pt', help='weights path')
    parser.add_argument('--save', default=f'./yolov9_wholebody_with_wheelchair_t.pt', type=str, help='save path')
    args = parser.parse_args()
    main(args)