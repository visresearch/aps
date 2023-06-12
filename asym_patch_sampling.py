import numpy as np


def get_overlap_region_box_x1y1x2y2(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1 = max(x1_1, x1_2)
    y1 = max(y1_1, y1_2)

    x2 = min(x2_1, x2_2)
    y2 = min(y2_1, y2_2)

    is_overlap = (x1 < x2) and (y1 < y2)

    if is_overlap:
        return x1, y1, x2, y2
    else:
        return None


def get_overlap_ratio_in_box1_x1y1x2y2(box1, box2):

    box_overlap = get_overlap_region_box_x1y1x2y2(box1, box2)

    ratio = 0.0

    if box_overlap is not None:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1, y1, x2, y2 = box_overlap

        area_b1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area_overlap = (x2 - x1) * (y2 - y1)

        ratio = area_overlap / area_b1

    return ratio


def get_boxs_non_overlap_ratio_x1y1x2y2(boxs, height, width, ngrid):
    ygrid_size = height / ngrid
    xgrid_size = width / ngrid

    num_grid = ngrid * ngrid
    ratios = np.ones((num_grid, ))

    for box in boxs:
        x1, y1, x2, y2 = box
        ids = []

        r1 = int(y1 / ygrid_size)
        r2 = int(y2 / ygrid_size)

        r1 = max(0, r1)
        r2 = min(r2, ngrid - 1)

        c1 = int(x1 / xgrid_size)
        c2 = int(x2 / xgrid_size)

        c1 = max(0, c1)
        c2 = min(c2, ngrid - 1)

        for r in range(r1, r2 + 1):
            tmp = r * ngrid
            y1 = r * ygrid_size
            y2 = y1 + ygrid_size
            for c in range(c1, c2 + 1):
                _id = tmp + c

                x1 = c * xgrid_size

                box_grid = (x1, y1, x1 + xgrid_size , y2)

                ratio = get_overlap_ratio_in_box1_x1y1x2y2(box_grid, box)

                ratios[_id] -= ratio
                if ratios[_id] < 1e-6:
                    ratios[_id] = 0.0

    return ratios


def get_boxs_non_overlap_x1y1x2y2(boxs, height, width, ngrid):
    ygrid_size = height / ngrid
    xgrid_size = width / ngrid

    num_grid = ngrid * ngrid
    ratios = np.ones((num_grid, ))

    for box in boxs:
        x1, y1, x2, y2 = box
        ids = []

        r1 = int(y1 / ygrid_size)
        r2 = int(y2 / ygrid_size)

        r1 = max(0, r1)
        r2 = min(r2, ngrid - 1)

        c1 = int(x1 / xgrid_size)
        c2 = int(x2 / xgrid_size)

        c1 = max(0, c1)
        c2 = min(c2, ngrid - 1)

        for r in range(r1, r2 + 1):
            tmp = r * ngrid
            for c in range(c1, c2 + 1):
                _id = tmp + c
                ratios[_id] = 0.0

    return ratios


def get_grid_boxs(height, width, ngrid, x_offset=0., y_offset=0.):

    ygrid_size = height / ngrid
    xgrid_size = width / ngrid

    boxs = []

    for i in range(ngrid):
        
        y1 = i * ygrid_size + y_offset
        y2 = y1 + ygrid_size

        for j in range(ngrid):
            x1 = j * xgrid_size + x_offset
            x2 = x1 + xgrid_size
            boxs.append((x1, y1, x2, y2))

    return boxs


def sample_overlap_less_patches_with_box1(box1, box2, ngrid, idxs_patch_box1 , power):
    # box1, box2: (i, j, h, w)
    y1, x1, h1, w1 = box1
    y2, x2, h2, w2 = box2

    n = len(idxs_patch_box1)

    boxs_patch = get_grid_boxs(h1, w1, ngrid, x1 - x2, y1 - y2)

    boxs_patch_select = [boxs_patch[i] for i in idxs_patch_box1]

    ratios = get_boxs_non_overlap_ratio_x1y1x2y2(boxs_patch_select, h2, w2, ngrid)
    ratios = ratios**power
    probs = ratios / np.sum(ratios)

    idxs_patch_box2 = np.random.choice(ngrid*ngrid, n, replace=False, p=probs)


    return idxs_patch_box2


def get_random_patch_sequence_index_pair(boxs1, boxs2, ngrid, sampling_ratio, time=4, power=3):
    npatch = ngrid * ngrid
    npatch_select = int(npatch * sampling_ratio)


    idx_patches = np.arange(npatch)
    idxs1=[]
    idxs2=[]
    np.random.shuffle(idx_patches)
    for i in range(time):
        idx1 = idx_patches[npatch_select*i:npatch_select*(i+1)]   
        idx2 = sample_overlap_less_patches_with_box1(boxs1, boxs2, ngrid, idx1, power)
        idxs1.append(idx1)
        idxs2.append(idx2)
    return idxs1, idxs2


