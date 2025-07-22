import numpy as np


def visualize_mask_cli(mask: np.ndarray, width: int=80, charset=" ░▒▓█"):
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("invalid input for visualization in cli")
    
    h, w = mask.shape
    aspect_ratio = h / w
    nh, nw = int(aspect_ratio * width * 0.5), width
    resized = np.resize(mask, (nw, nh * 2))
    
    levels = len(charset) - 1

    for i in range(0, resized.shape[0] - 1, 2):
        top_row = resized[i]
        bottom_row = resized[i + 1]
        line = ""
        for top, bottom in zip(top_row, bottom_row):
            # print(top, bottom)
            t, b = int(top // (255 / levels)), int(bottom // (255 / levels))
            # print("t:", top, levels, t)
            # print("b:", bottom, levels, b)
            top_char = charset[t]
            bottom_char = charset[b]
            avg = (int(top) + int(bottom)) // 2
            line += charset[int(avg * levels // 255)]
        print(line)

charset1 = ".:-=+*#%@"
charset2 = " ░▒▓█"
charset3 = " ▁▂▃▄▅▆▇█"
charset4 = " ▏▎▍▌▋▊▉█"

test_image = np.tile(np.linspace(0, 255, 40, dtype=np.uint8), (20, 1))
visualize_mask_cli(test_image, 
                   charset=charset2
                   )
print(test_image.shape)