import numpy as np
import cv2
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from skimage.transform import warp, SimilarityTransform
from skimage.measure import ransac
from tqdm import tqdm
from scipy.optimize import leastsq
from astropy.io import fits
from scipy.signal import medfilt2d
from scipy.signal import fftconvolve
from scipy.optimize import least_squares

func = SimilarityTransform
# func = EuclideanTransform

def fitswrite(fileout, im, header=None):
    from astropy.io import fits
    import os
    if os.path.exists(fileout):
        os.remove(fileout)
    if header is None:
        fits.writeto(fileout, im, output_verify='fix', overwrite=True, checksum=False)
    else:
        fits.writeto(fileout, im, header, output_verify='fix', overwrite=True, checksum=False)


def fitsread(filein):
    head = '  '
    hdul = fits.open(filein)

    try:
        data0 = hdul[0].data.astype(np.float32)
        head = hdul[0].header
    except:
        hdul.verify('silentfix')
        data0 = hdul[1].data
        head = hdul[1].header

    return data0, head


def checkdata(filelist, Flip):
    im = []
    header = []
    files = []
    for file in tqdm(filelist):
        try:
            imorg, hd = fitsread(file)  # 读取fits 文件
            # imorg=removestrip(imorg) #原始信号imorg消除cmos条纹

            if Flip == 0:
                imorg = np.flip(imorg, 0)  # 上下镜像反转
            im.append(imorg)
            header.append(hd)
            files.append(file)
        except:
            continue

    im = np.array(im).astype('int16')
    return im, header, files


def showim(im, k=3, cmap='gray'):
    mi = np.max([im.min(), im.mean() - k * im.std()])
    mx = np.min([im.max(), im.mean() + k * im.std()])
    if len(im.shape) == 3:
        plt.imshow(im, vmin=mi, vmax=mx, origin='lower')
    else:
        plt.imshow(im, vmin=mi, vmax=mx, cmap=cmap, interpolation='bicubic', origin='lower')
    plt.imsave('input1.png', im, cmap='gray', vmin=mi, vmax=mx)
    return


def disk(M, N, r0):
    X, Y = np.meshgrid(np.arange(int(-(N / 2)), int(N / 2)), np.linspace(-int(M / 2), int(M / 2) - 1, M))
    r = (X) ** 2 + (Y) ** 2
    r = (r ** 0.5)
    im = r < r0  # 半径内的点位true
    return im


# 定义相似变换模型
def transform(params, src):
    # 提取参数
    s, theta, tx, ty = params
    # 计算旋转矩阵
    rotation_matrix = np.array([[np.cos(theta), np.sin(theta)],
                                [-np.sin(theta), np.cos(theta)]])
    # 应用缩放和旋转
    transformed = np.dot(src, rotation_matrix) * np.array([s, s]) + np.array([tx, ty])  # src[x,y]右乘旋转矩阵，所以旋转矩阵内sin的符号改变
    return transformed


# 定义残差函数
def residuals_(params, src, dst):
    return np.ravel(transform(params, src) - dst)


# 定义残差函数2
def residuals_2(params, src, dst):
    return (transform(params, src) - dst)


def zscore2(im):  # 图像标准化
    im = (im - np.mean(im)) / im.std()
    return im


def immove2(im, dx=0, dy=0):  # 图像平移
    im2 = im.copy()
    tform = SimilarityTransform(translation=(dx, dy))
    im2 = warp(im2, tform.inverse, output_shape=(
        im2.shape[0], im2.shape[1]), mode='constant', cval=0, preserve_range=True)

    return im2


def removenan(im, key=0):
    """
    remove NAN and INF in an image
    """
    im2 = np.copy(im)
    arr = np.isnan(im2)
    im2[arr] = key
    arr2 = np.isinf(im2)
    im2[arr2] = key

    return im2


def removeray(im, T=0.2):
    c = medfilt2d(im, 3)
    # d=np.abs(removenan(im/c)-1)>T
    d = np.abs(removenan(im - c)) > (T * c)
    out = c * d + (1 - d) * im
    return out, d


def center_im(im, header, mode):  #

    ##根据头文件平移到中心

    dx_1 = header['CRPIX1'] - (header['NAXIS1'] + 1) / 2
    dy_1 = header['CRPIX2'] - (header['NAXIS2'] + 1) / 2

    print(dx_1, dy_1)

    im = immove2(im, -dx_1, -dy_1)

    return im


# 计算雅可比矩阵
def jacobian(params, src, dst):
    s, theta, tx, ty = params
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    J = np.zeros((2 * len(dst), len(params)))

    for i in range(len(src)):
        x, y = src[i]
        transformed_x = s * (np.cos(theta) * x - np.sin(theta) * y) + tx
        transformed_y = s * (np.sin(theta) * x + np.cos(theta) * y) + ty

        # 对每个参数的偏导数
        J[2 * i, 0] = (np.cos(theta) * x - np.sin(theta) * y)
        J[2 * i, 1] = s * (-np.sin(theta) * x - np.cos(theta) * y)
        J[2 * i, 2] = 1
        J[2 * i, 3] = 0

        J[2 * i + 1, 0] = (np.sin(theta) * x + np.cos(theta) * y)
        J[2 * i + 1, 1] = s * (np.cos(theta) * x - np.sin(theta) * y)
        J[2 * i + 1, 2] = 0
        J[2 * i + 1, 3] = 1

    return J


# 光流对齐2
def align_opflow_2(im1, im2, winsize=11, step=5, r_t=2, arrow=0, sample=10, mask=1):  # , mask=0):

    # mask = np.ones(im1.shape) > 0
    # mask=(im1>im1.mean())+(im2>im2.mean())
    # mask=mask>0
    flow = cv2.calcOpticalFlowFarneback(im1, im2, flow=None, pyr_scale=0.5,
                                        levels=5, winsize=winsize, iterations=10, poly_n=5, poly_sigma=1.1, flags=0)

    # calculate_translation_rotation(flow)
    # flow = flow * mask[:, :, np.newaxis]
    # 按照step 选取有效参考点
    # mask = (im1  > im1.mean())+(im2 > im2.mean())
    # mask = mask > 0
    h, w = im1.shape
    x1, y1 = np.meshgrid(np.arange(w), np.arange(h))
    x2 = x1.astype('float32') + flow[:, :, 0]
    y2 = y1.astype('float32') + flow[:, :, 1]
    x1 = x1[mask][::step]
    y1 = y1[mask][::step]
    x2 = x2[mask][::step]
    y2 = y2[mask][::step]

    src = (np.vstack((x1.flatten(), y1.flatten())).T)  # 源坐标
    dst = (np.vstack((x2.flatten(), y2.flatten())).T)  # 目标坐标

    # 抽样
    indices = sample_points_evenly(src, dst, sample, 1)
    src = src[indices]
    dst = dst[indices]

    s = dst - src  # 位移量
    # print(np.shape(s))
    # print(np.mean(s[0]),np.mean(s[1]),np.mean(s))

    Dlt0 = ((np.abs(s[:, 0]) > 0) * 1.0 + (np.abs(s[:, 1]) > 0)) > 0  # 判断是否无位移

    if Dlt0.sum() > 0:  # 处理有位移的图像
        dst = dst[Dlt0]
        src = src[Dlt0]
        s = s[Dlt0]
        #####筛选有效参考点################

        # 如果要考虑旋转，可以使用这个函数。但旋转也会带来更大的累计误差。慎重使用。同时返回量是一个齐次矩阵。代码要改很多
        model, D = ransac((src, dst), func, min_samples=4,
                          residual_threshold=r_t, max_trials=200)
        # print(np.shape(D),D.sum())

        d = [model.scale, model.rotation, model.translation[0], model.translation[1]]

        # 计算残差
        residuals = model.residuals(src[D], dst[D])
        # print(np.mean(residuals))
        sq = [x ** 2 for x in residuals]
        sigma2 = sum(sq) / (D.sum() - 4)
        # print('残差的方差',sigma2)

        # x,y方向残差
        rv = residuals_2(d, src[D], dst[D])
        # print(np.mean(rv[:,0]))
        sq1 = [x ** 2 for x in rv[:, 0]]
        xsigma2 = sum(sq1) / (D.sum() - 4)
        # print('x残差的方差',xsigma2)

        sq2 = [x ** 2 for x in rv[:, 1]]
        ysigma2 = sum(sq2) / (D.sum() - 4)
        # print('y残差的方差',ysigma2)

        rsigma2 = (sum(sq1) + sum(sq2)) / (D.sum() - 4)
        # print('r残差的方差',rsigma2)

        # 初始参数 [s, theta, tx, ty]
        initial_params = [1, 0, 0, 0]
        # 最小二乘拟合
        result = leastsq(residuals_, initial_params, args=(src[D], dst[D]), full_output=True)
        model1 = result[0]
        cov1 = result[1]
        # print('最小二乘非线性拟合',model1)
        #  print('协方差矩阵',cov1)

        err = [np.sqrt(cov1[0, 0] * sigma2), np.sqrt(cov1[1, 1] * sigma2) / np.pi * 180 * 3600,
               np.sqrt(cov1[2, 2] * xsigma2), np.sqrt(cov1[3, 3] * ysigma2)]
        # print('四项误差s,theta,dx,dy',err)

        ###########如果需要，画光流场##################
        '''if arrow == 1:
            plt.figure()
            showim(im1)
            x = src[D, 0]
            y = src[D, 1]
            fx = s[D, 0]
            fy = -s[D, 1]
            plt.quiver(x, y, fx, fy, color='r', scale=0.2,
                       scale_units='dots', minshaft=2)'''

        try:
            flag = D.sum() / Dlt0.sum()  # 有效控制点占比， 用于评价配准的概率
            d = [model.scale, model.rotation, model.translation[0], model.translation[1]]  #
        except:
            flag = -999
            d = [0, 0, 0, 0]

        ###############雅可比算误差
        # 提取最优参数
        optimal_params = d
        # 计算雅可比矩阵
        J = jacobian(optimal_params, src[D], dst[D])

        # 协方差矩阵计算
        residuals_val = residuals_(optimal_params, src[D], dst[D])
        # print(np.shape(residuals_val))
        # print('方差', np.sum(residuals_val ** 2) / (len(dst[D]) - len(optimal_params)))
        cov_matrix = np.linalg.inv(J.T @ J)
        err = [np.sqrt(cov_matrix[0, 0] * sigma2), np.sqrt(cov_matrix[1, 1] * sigma2) / np.pi * 180 * 3600,
               np.sqrt(cov_matrix[2, 2] * xsigma2), np.sqrt(cov_matrix[3, 3] * ysigma2)]
        print('UABJM err(s,theta,dx,dy):', err)

    return d, model, flag, flow, err


# 定义一个函数来进行均匀抽样
def sample_points_evenly(points, dst, grid_size=10, num=1):
    # 计算点集的边界
    delt = dst - points
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])

    # 创建网格中心点
    x_grid = np.linspace(min_x, max_x, grid_size)
    y_grid = np.linspace(min_y, max_y, grid_size)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    grid_centers = np.column_stack([x_mesh.ravel(), y_mesh.ravel()])

    # 计算每个点到每个网格中心的距离
    distances = cdist(points, grid_centers)

    # 对于每个点，找到最近的网格中心
    nearest_grid_idx = np.argmin(distances, axis=1)

    # 从每个非空的网格中随机选择一个点
    points_per_grid = np.array([np.where(nearest_grid_idx == i)[0] for i in range(grid_size * grid_size)], dtype=object)

    sampled_indices = []
    for idx_points in points_per_grid:
        if idx_points.size > 0:
            for _ in range(num):
                r = (delt[idx_points] ** 2).sum(axis=1)
                r_idx = np.abs(r - np.median(r)).argmin()
                sampled_idx = idx_points[r_idx]
                # sampled_idx = np.random.choice(idx_points)

                sampled_indices.append(sampled_idx)

    return np.array(sampled_indices)


# 亚像素的互相关
def cc_align(im1, im2):
    def sub_location(cor_matrix):
        """
        通过二次拟合确定亚像素精度的峰值位置。
        """
        # 找到互相关矩阵的最大值位置
        max_idx = np.unravel_index(np.argmax(cor_matrix), cor_matrix.shape)

        # 提取峰值周围的3x3区域
        y, x = max_idx
        if x > 0 and x < cor_matrix.shape[1] - 1 and y > 0 and y < cor_matrix.shape[0] - 1:
            patch = cor_matrix[y - 1:y + 2, x - 1:x + 2]
        else:
            return max_idx

        # 二次拟合
        def q_fit(params, x, y, z):
            a, b, c, d, e, f = params
            return a * x ** 2 + b * y ** 2 + c * x * y + d * x + e * y + f - z

        x_coords = np.arange(-1, 2)
        y_coords = np.arange(-1, 2)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        z_values = patch.flatten()
        z_values = z_values - np.mean(z_values)

        initial_guess = [1, 1, 0, 0, 0, 0]
        result = least_squares(q_fit, initial_guess, args=(x_grid.flatten(), y_grid.flatten(), z_values), max_nfev=1000,
                               ftol=1e-6, xtol=1e-6)
        a, b, c, d, e, f = result.x

        # 计算亚像素精度的偏移量
        subpixel_x = -d / (2 * a)
        subpixel_y = -e / (2 * b)
        print('y,x', y, x, subpixel_y, subpixel_x)

        return (2048 - y - subpixel_y, 2048 - x - subpixel_x)

    im1 = zscore2(im1)
    im2 = zscore2(im2)
    # 计算互相关矩阵
    cor_matrix = fftconvolve(im1, im2[::-1, ::-1], mode='same')

    # 找到互相关矩阵的峰值位置
    location = sub_location(cor_matrix)

    # 返回亚像素精度的偏移量
    return location

def c_plot(im,s,c,back,gain):
    C_map=np.zeros(im.shape)
    avg=im[s].mean()
    rms=im[s].std()
    C_map[0]=np.sqrt(np.maximum((im[s]-1*avg)/rms+back[0],0))/gain
    C_map[2]=np.sqrt(np.maximum((im[c]-1*avg)/rms+back[1],0))/gain
    C_map[1]=C_map[2]
    C_map=C_map.transpose(1,2,0)
    '''
    plt.figure()
    plt.imshow(C_map)
    plt.title(c)'''

    return C_map

def region_show(im):
    for i in range(3):
        im[i] = medfilt2d(im[i], 5)
    # im=im[:,600:1100,600:1100]

    D = zscore2(im[0]) < 3
    p = np.polyfit(im[0, D], im[2, D], 2)
    im[0] = np.polyval(p, im[0])
    back = [0.3, 0.4]
    gain = np.sqrt(4)

    map1 = c_plot(im, 0, 1, back, gain)
    map2 = c_plot(im, 0, 2, back, gain)

    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot2grid((1, 2), (0, 0), colspan=1, rowspan=1)
    ax1.imshow(map1, origin='lower')
    ax1.set_title("Coarse alignment")
    ax1.tick_params(axis='both', which='major')

    ax2 = plt.subplot2grid((1, 2), (0, 1), colspan=1, rowspan=1)
    ax2.imshow(map2, origin='lower')
    ax2.set_title("Fine alignment")
    ax2.tick_params(axis='both', which='major')

    return
