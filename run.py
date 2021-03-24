
import config


def run(model, image):
    image = config.IMAGE_TRANFORM_INFERENCE(image)
    image = image.unsqueeze(0)
    image = image.to(config.DEVICE)

    with torch.no_grad():
        model.eval()
        output = model(image)
        predict = torch.argmax(output, -1).item()

    return predict


if __name__ == '__main__':

    import argparse
    from PIL import Image
    import model
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    args = parser.parse_args()

    net = model.WasteNet().to(config.DEVICE)
    net.load_state_dict(torch.load(args.weights, map_location=config.DEVICE))

    image = Image.open(args.image)
    image_copy = image.copy()
    image.close()

    score = run(net, image_copy)

    if score > 0.5:
        print('It\'s a hot dog. {:.3}'.format(score))
    else:
        print('It\'s NOT a hot dog. {:.3}'.format(score))
