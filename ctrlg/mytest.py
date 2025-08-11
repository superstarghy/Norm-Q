import torch
# import matplotlib.pyplot as plt


def asymq(t, qmin, qmax):
    tmax = torch.max(t)
    tmin = torch.min(t)
    scale = (tmax - tmin) /  (qmax - qmin)
    zero_point = qmin - torch.round(tmin / scale)
    # ft = torch.clip(torch.round(t / scale + zeropoint), qmin, qmax)
    return scale, zero_point


def symq(t, qmax, ratio=1):
    tmax = torch.max(torch.abs(t))
    scale = tmax / qmax
    zero_point = 0
    return scale, zero_point


def q_matmul(A, B, b=16):
    # symmetric quantization
    r = 2 ** b
    scale1 = 1 / (r - 1)
    qA = torch.clamp(torch.ceil(A / scale1), 0, r-1)
    scale2 = 1 / (r - 1)
    qB = torch.clamp(torch.ceil(B / scale2), 0, r-1)
    qC = torch.matmul(qA, qB)
    qC.mul_(scale1 * scale2)

    # 1/x scale, f(x)=-1/x+2b
    # r = 2 ** b
    # qA = torch.clamp(torch.ceil(r - 1/A), 0, r-1)
    # qB = torch.clamp(torch.ceil(r - 1/B), 0, r-1)
    # qC = torch.matmul(qA, qB)
    

    # A = torch.quantize_per_tensor(A, scale, zero_point, dtype=torch.quint8)
    # B = torch.quantize_per_tensor(B, scale, zero_point, dtype=torch.qint8)
    # fc = torch.ao.nn.quantized.Linear(A.shape[1], B.shape[0], bias_=False, dtype=torch.qint8)
    # fc.set_weight_bias(B.T, None)
    # C = fc(A)
    # torch.ao.nn.quantized.modules.linear.Linear
    # C = torch.ao.nn.quantized.functional.linear(A, B.T) # not supported for some GPU
    # C.dequantize()

    return qC


def q_matmul_log(A, B, b=16):
    bd = len(B.shape) - 2
    A_max = torch.amax(A, dim=-1, keepdim=True)
    B_max = torch.amax(B, dim=bd, keepdim=True)
    A = A - A_max
    B = B - B_max
    A.exp_()
    B.exp_()

    C = q_matmul(A, B, b)
    C.log_()
    C.add_(A_max + B_max)

    return C


def q_matmul_loga_b(A, B, b=16):
    A_max = torch.amax(A, dim=-1, keepdim=True)
    A = A - A_max
    A.exp_()
    C = q_matmul(A, B, b)
    C.log_()
    C.add_(A_max)

    return C


def q_matmul_a_logb(A, B, b=16):
    bd = len(B.shape) - 2
    B_max = torch.amax(B, dim=bd, keepdim=True)
    B = B - B_max
    B.exp_()
    C = q_matmul(A, B, b)
    C.log_()
    C.add_(B_max)

    return C


class _test_matmul_log:
    bit_width = 12
    count = 0
    sum_scale_A = 0
    sum_scale_B = 0
    sum_zero_point_A = 0
    sum_zero_point_B = 0

    @staticmethod
    def matmul_log(A, B):
        _test_matmul_log.count += 1
        
        bd = len(B.shape) - 2
        A_max = torch.amax(A, dim=-1, keepdim=True)
        B_max = torch.amax(B, dim=bd, keepdim=True)
        A = A - A_max
        B = B - B_max

        A.exp_()
        B.exp_()
        # C = torch.matmul(A, B)

        # quantization
        r = 2 ** _test_matmul_log.bit_width
        factor = 1

        # scale_A = torch.max(A) / (r - 1) * factor
        # scale_A = 1 / (r-1)
        # zero_point_A = 0
        # qA = torch.clamp(torch.ceil(A / scale_A) + zero_point_A, 0, r-1)
        qA = torch.clamp(torch.ceil(r - A.reciprocal()), 0, r-1)
        # _test_matmul_log.sum_scale_A += scale_A
        # _test_matmul_log.sum_zero_point_A += zero_point_A
        # print(f"{_test_matmul_log.count}, Amin={torch.min(A)}, Amax={torch.max(A)}, \
        #     S={scale_A}, ZP={zero_point_A}, \
        #     avgS={_test_matmul_log.sum_scale_A / _test_matmul_log.count} \
        #     avgZP={_test_matmul_log.sum_zero_point_A / _test_matmul_log.count}")

        # scale_B = torch.max(B) / (r - 1) * factor
        # scale_B = 1 / (r-1)
        # zero_point_B = 0
        # qB = torch.clamp(torch.ceil(B / scale_B) + zero_point_B, 0, r-1)
        qB = torch.clamp(torch.ceil(r - B.reciprocal()), 0, r-1)
        # scale_B = (torch.log(tmax) - torch.log(tmin)) / (r - 1)
        # zero_point_B = (r - 1) - torch.round(torch.log(tmax) / scale_B)
        # B = torch.round(torch.log(B) / scale_B) + zero_point_B 
        # B = r - torch.round(torch.clamp(1 / B, 1, r))

        qC = torch.matmul(qA, qB)
        zero_point = torch.mean(qC - torch.matmul(A, B).reciprocal())
        _test_matmul_log.sum_zero_point_A += zero_point
        print(f"{_test_matmul_log.count}, ZP={zero_point}, \
            avgZP={_test_matmul_log.sum_zero_point_A / _test_matmul_log.count}")
        # r = 2 ** 16
        # scale1, zero_point1 = symq(A, r/2-1)
        # qA = torch.clamp(torch.round(A / scale1), -r/2, r/2-1)
        # A = torch.quantize_per_tensor(A, scale, zero_point, dtype=torch.quint8)
        # scale2, zero_point2 = symq(B, r/2-1)
        # qB = torch.clamp(torch.round(B / scale2), -r/2, r/2-1)
        # qC = torch.matmul(qA, qB)
        # qC.mul_(scale1 * scale2)

        # demonstrate distribution of quantized A, B, C
        # bins = 1024
        # plt.figure(figsize=(12,4))
        # plt.subplot(131)
        # plt.hist(torch.flatten(qA.clone()).cpu(), bins=bins, edgecolor='black')
        # plt.title("qA%d, min=%.2f, max=%.2f" % (_test_matmul_log.count, torch.min(qA), torch.max(qA)))
        # plt.ylim(0, 30000)
        # plt.xscale('log')

        # plt.subplot(132)
        # plt.hist(torch.flatten(qB.clone()).cpu(), bins=bins, edgecolor='black')
        # plt.title("qB%d, min=%.2f, max=%.2f" % (_test_matmul_log.count, torch.min(qB), torch.max(qB)))
        # plt.ylim(0, 2e6)
        # plt.xscale('log')

        # plt.subplot(133)
        # plt.hist(torch.flatten(qC.clone()).cpu(), bins=bins, edgecolor='black')
        # plt.title("qC%d, bin=%d, min=%.4f, max=%.4f" % (_test_matmul_log.count, bins, torch.min(qC), torch.max(qC)))
        # plt.ylim(0, 3e5)
        # plt.xscale('log')

        # plt.tight_layout()
        # plt.savefig(f'fig2/{_test_matmul_log.count}.png')
        # plt.close()

        # qC.mul_(scale_A * scale_B) # dequantize
        qC = (qC-zero_point).reciprocal_()
        qC.log_()
        # qC.nan_to_num_(neginf=-1e30)

        # C.log_()
        qC.add_(A_max + B_max)

        return qC

