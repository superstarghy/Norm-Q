import torch
import torch.nn.functional as F

def kl_divergence(P, Q, epsilon=1e-10):
    return F.kl_div(torch.log(Q + epsilon), P + epsilon, reduction="batchmean")

def bceil(t, bitwidth=8):
    cluster = 2 ** bitwidth
    return torch.ceil(t * cluster).clamp(0, cluster - 1).div(cluster)

def bfloor(t, bitwidth=8):
    cluster = 2 ** bitwidth
    return torch.floor(t * cluster).clamp(0, cluster - 1).div(cluster)

def bround(t, bitwidth=8):
    cluster = 2 ** bitwidth
    return torch.round(t * cluster).clamp(0, cluster - 1).div(cluster)


def branch_list(p, bitwidth):
    bf = bfloor(p, bitwidth)
    bc = bceil(p, bitwidth)
    if p == 0.0:
        blist = [0.0, ]
    elif bf == bc:
        blist = [bf, ]
    else:
        # l = 2 ** bitwidth
        # blist = [i/l for i in range(l)]
        blist = [bf, bc]
    return blist

def quantize_bnb(Pz, bitwidth=8, epsilon=1e-10):
    '''
    Quantization using branch and bound strategy
    '''
    # prune zeros
    nonzero_indices = torch.nonzero(Pz).flatten()
    P = Pz[nonzero_indices]

    Q = torch.zeros_like(P)
    print(f"Complexity: (2^{bitwidth}) ^ {len(Q)}")
    upper_bound = torch.inf
    min_loss = torch.inf
    min_Q = torch.zeros_like(P)

    Sp = torch.sum(P).item()
    Sq = 1
    runstack = []

    

    # root
    # while(level)
    # for level in range(len(P)):
    #     p = P[level]
    #     blist = branch_list(p, bitwidth)
    #     Sp -= p
    #     for q in blist:
    #         Sq -= q
    #         Sq += q

    def branch_and_bound_min(P, Q, Sp, Sq, level):
        nonlocal bitwidth, upper_bound, min_loss, min_Q
        if level == 0:  # leaf
            if Sq != 0:  # unsatisfied, sum Q < 1
                return 0
            else: # a feasible solution, sum Q = 1
                loss = kl_divergence(P, Q, epsilon)
                if loss < min_loss:
                    min_loss = loss
                    min_Q = torch.clone(Q)
                    upper_bound = loss
                return 1

        if Sq < 0:  # unsatisfied, sum Q > 1
            return 0
        for i in range(level):
            Q[i] = P[i] * Sq / Sp
        loss = kl_divergence(P, Q, epsilon)
        if loss >= upper_bound:  # pruning
            return 0
        
        p = P[level - 1]
        blist = branch_list(p, bitwidth)
        flag = 0
        for q in blist:
            Q[level - 1] = q 
            if branch_and_bound_min(P, Q, Sp-p, Sq-q, level-1):
                flag = 1
        
        return flag

    branch_and_bound_min(P, Q, torch.sum(P).item(), 1, len(P))

    Q = torch.zeros_like(Pz)
    Q[nonzero_indices] = min_Q
    return Q, min_loss


if __name__ == "__main__":
    bitwidth = 2
    tmp = torch.rand([2,7])
    # tmp.div_(torch.sum(tmp, dim=-1, keepdim=True))
    print(tmp)
    for t in range(tmp.size(0)):
        tmp[t], _ = quantize_bnb(tmp[t], bitwidth)
    print(tmp)
    
    tmp = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0, 0.2, 0.04, 0.06, 0.1, 0.1])
    tmp = torch.tensor([0.9946, 0.4991, 0.2353, 0.6236, 0.8811, 0.1389, 0.1313])
    tmp = torch.tensor([0, 1/3, 0, 1/3, 1/6, 1/6])
    Q_, loss_ = quantize_bnb(tmp, bitwidth)
    print(Q_, loss_)
    print(Q_*(2 ** bitwidth))
    print(tmp * (2 ** bitwidth))
