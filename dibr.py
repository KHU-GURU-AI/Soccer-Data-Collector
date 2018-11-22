import mxnet as mx
import numpy as np
import data
import argparse
import cv2


def dibr(model_path, ctx, fname=None):
    n_batch = 50
    batch_size = 10
    start = 0  # 50*24*60/batch_size + 1
    #scale = (-15, 17)
    #data_frames = 1
    #flow_frames = 0
    upsample = 4
    data_shape = (384, 160)
    udata_shape = (data_shape[0] * upsample, data_shape[1] * upsample)
    base_shape = (432 * upsample, 180 * upsample)
    pos = [(24, 10), (0, 0), (0, 20), (48, 0), (48, 20)]
    norm = np.zeros((3, base_shape[1], base_shape[0]), dtype=np.float32)
    boarder = 32 * upsample
    feather = np.ones((udata_shape[1], udata_shape[0]), dtype=np.float32)
    for i in range(udata_shape[1]):
        for j in range(udata_shape[0]):
            feather[i, j] = min((min(i, j) + 1.0) / boarder, 1.0)
    for p in pos:
        up = (p[0] * upsample, p[1] * upsample)
        norm[:, up[1]:up[1] + udata_shape[1], up[0]:up[0] + udata_shape[0]] += feather
    # 결론은 이 부분은 그냥 오른쪽용 이미지를 받아오는 부분 근데 뭐이리 많이 필요해;;
    test_data = cv2.imread("test.png")
    """
    test_data = data.Mov3dStack('영상 경로', data_shape, batch_size, scale, output_depth=False,
                                data_frames=data_frames, flow_frames=flow_frames,
                                test_mode=True, source=source, upsample=upsample, base_shape=base_shape)
    """
    # 날리고 이미지 받아오는 부분으로 바꿔야할듯
    #
    # 이 부분이 중요
    init = mx.init.Load(model_path, verbose=True)
    model = mx.model.FeedForward(ctx=ctx, symbol='depth image', initializer=init)
    model._init_params(dict(test_data.provide_data + test_data.provide_label))

    data_names = [x[0] for x in test_data.provide_data]
    model._init_predictor(test_data.provide_data)
    data_arrays = [model._pred_exec.arg_dict[name] for name in data_names]

    for i in range(n_batch):
        P = np.zeros((batch_size,) + norm.shape, dtype=np.float32)
        for p in pos:
            test_data.seek(start + i)
            test_data.fix_p = p
            batch = test_data.next()
            mx.executor._load_data(batch, data_arrays)
            model._pred_exec.forward(is_train=False)

            up = (p[0] * upsample, p[1] * upsample)
            P[:, :, up[1]:up[1] + udata_shape[1], up[0]:up[0] + udata_shape[0]] += model._pred_exec.outputs[
                                                                                       0].asnumpy() * feather

        P = np.clip(P / norm, 0, 255)

        P = P.astype(np.uint8).transpose((0, 2, 3, 1))

        save_image = mx.ndarray.save(fname, P)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert movie to 3D')
    parser.add_argument('--model_path', type=str, default='exp/right', help='Path to model parameters file')
    parser.add_argument('--ctx', type=int, default=0, help='GPU id to run on')
    #parser.add_argument('--source', type=str, default='test_idx', help='Source index prefix')
    parser.add_argument('--output', type=str, default='/data/output', help='Output prefix')
    args = parser.parse_args()

    dibr(args.model_path, mx.gpu(args.ctx), args.output)
