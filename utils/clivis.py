import numpy as np


def visualize_mask_cli(mask: np.ndarray, charset=" ░▒▓█"):
    """
    Visualize 2D uint8 image in terminal
    """
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("invalid input for visualization in cli")
    
    levels = len(charset) - 1
    for i in range(0, mask.shape[0] - 1, 2):
        top_row = mask[i]
        bottom_row = mask[i + 1]
        line = ""
        for top, bottom in zip(top_row, bottom_row):
            # print(top, bottom)
            t, b = int(top // (255 / levels)), int(bottom // (255 / levels))
            # print("t:", top, levels, t)
            # print("b:", bottom, levels, b)
            # top_char = charset[t]
            # bottom_char = charset[b]
            avg = (int(top) + int(bottom)) // 2
            line += charset[int(avg * levels // 255)]
        print(line)


if __name__ == "__main__":
    charset1 = ".:-=+*#%@"
    charset_smooth = " ░▒▓█"
    charset3 = " ▁▂▃▄▅▆▇█"
    charset4 = " ▏▎▍▌▋▊▉█"
    
    H, W = 20, 40
    test_image = np.tile(np.linspace(0, 255, W, dtype=np.uint8), (H, 1))
    visualize_mask_cli(test_image, charset=charset_smooth)
    print(test_image.shape)