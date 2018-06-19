# cython: language_level=3
import math

cimport cython
import bisect

import numpy as np
cimport numpy as np
from libc.math cimport ceil

bisect_left = bisect.bisect_left
bisect_right = bisect.bisect_right

from cython.view cimport array
import random

__version__ = '0.1.0'


beta = np.random.beta

cdef packed struct LinkedNode:
    int next
    int val
    int round

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef feerci(np.ndarray[np.float64_t,ndim=1] impostors,np.ndarray[np.float64_t,ndim=1] genuines, is_sorted=False,int m=10000, ci=0.95):
    """
    O(m log n) function for EER and non-parametric bootstrapped confidence bound.
    Uses beta distribution to estimate ranks at specifically chosen points by a binary search comparable to FEER's. Returns
    eer, estimated eers and ci lower and upper bound. If m <= 0, only calculates EER and returns an empty list and None
    for lower and upper bound.
    :param impostors: list of impostor scores 
    :param genuines: list of genuine scores
    :param is_sorted: whether lists are sorted or not (will sort if not)
    :param m: amount of bootstrap iterations to run
    :param ci: confidence interval to calculate, 0 = 0%, 1= 100%.
    :return: EER, Bootstrapped_Eers, ci_lower_bound, ci_upper_bound
    """

    if not is_sorted:
        impostors.sort(axis=0)
        genuines.sort(axis=0)
    cdef float eer = feer(impostors,genuines,is_sorted=True)
    if m <= 0:
        return eer,[], None, None

    # Initialize used variables
    cdef int genlen,implen,d_implen,d_genlen,i,i_m
    cdef int gomin,gomax, gbmin, gbmax
    cdef int iomin,iomax, ibmin, ibmax
    cdef int rmax,rmin,rmid,imax,imin,imid
    genlen = genuines.shape[0]
    implen = impostors.shape[0]
    d_genlen = genlen - 1
    d_implen = implen - 1
    cdef int kg1,kg2,kp1,kp2,ip1,ip2,ig1,ig2
    cdef int imp_ll_min,gen_ll_min,imp_ll_max,gen_ll_max,centre_offset
    cdef float frr_1,frr_2,far_1, vmid, sp1,sp2,sg1,sg2

    cdef float[:] eers = array((m,), itemsize=sizeof(float),format='f')
    cdef LinkedNode[:] imp_bs = array((implen,),itemsize=sizeof(LinkedNode),format='iii')
    cdef LinkedNode[:] gen_bs = array((genlen,),itemsize=sizeof(LinkedNode),format='iii')

    # Initialize lists. Loop over length that imp & gen list have in common first
    # This is an O(n) operation that is amortized across the multiple runs of the bootstrap
    for i in range(min(genlen,implen)):
        gen_bs[i].next = -1
        gen_bs[i].round = -1
        imp_bs[i].next = -1
        imp_bs[i].round = -1
    # Now loop over remaining part
    if genlen > implen:
        for i in range(implen,genlen):
            gen_bs[i].next = -1
            gen_bs[i].round = -1
    elif implen > genlen:
        for i in range(genlen,implen):
            imp_bs[i].next = -1
            imp_bs[i].round = -1

    for i_m in range(m):
        # Initialize interest ranges. One for the sampled set, one for the bootstrapped set.
        gomin, gomax = gbmin, gbmax = (0, d_genlen)
        iomin, iomax = ibmin, ibmax = (0, d_implen)

        # Initialize current linked list ranges (if -1, no higher/lower ll node exists
        gen_ll_min = gen_ll_max = imp_ll_min = imp_ll_max = -1
        head_gen_ll = (gbmax + gbmin) // 2
        head_imp_ll = (ibmax + ibmin) // 2

        centre_offset = 0
        while gbmax - gbmin > 1 or ibmax - ibmin > 1:

            # Pick indices in the middle of the current range of interest (on the bootstrap set)
            kg1 = (gbmax + gbmin) // 2
            kp1 = (ibmax + ibmin) // 2
            # Pick the ones next to it as well
            kg2 = kg1 + 1
            kp2 = kp1 + 1

            # Draw samples from the beta dists. If specific sample has already been draw, do not draw it, but update the linked list reference.

            if gen_bs[kg1].round != i_m:
                gen_bs[kg1].next = kg2
                gen_bs[kg1].val = round(beta(kg1 - gbmin - centre_offset +1,gbmax- kg1 +1) * (gomax - gomin) + gomin)
                gen_bs[kg1].round = i_m
                if gen_ll_min != -1:
                    gen_bs[gen_ll_min].next = kg1
            else:
                gen_bs[kg1].next = kg2

            if gen_bs[kg2].round != i_m:
                gen_bs[kg2].next = gen_ll_max
                gen_bs[kg2].val = round(beta(1,gbmax - kg1+1) * (gomax -  gen_bs[kg1].val) +  gen_bs[kg1].val)
                gen_bs[kg2].round = i_m

            if imp_bs[kp1].round != i_m:
                imp_bs[kp1].next = kp2
                imp_bs[kp1].val = round(beta(kp1 - ibmin - centre_offset +1,ibmax- kp1 +1) * (iomax - iomin) + iomin)
                imp_bs[kp1].round = i_m
                if imp_ll_min != -1:
                    imp_bs[imp_ll_min].next = kp1
            else:
                imp_bs[kp1].next = kp2

            if imp_bs[kp2].round != i_m:
                imp_bs[kp2].next = imp_ll_max
                imp_bs[kp2].val = round(beta(1,ibmax - kp1+1) * (iomax -  imp_bs[kp1].val) +  imp_bs[kp1].val)
                imp_bs[kp2].round = i_m


            ig1 = gen_bs[kg1].val
            ip2 = d_implen - imp_bs[kp1].val

            # Check if genuine score range lies higher than impostor score range
            if genuines[ig1] > impostors[ip2]:
                gbmax = gen_ll_max = kg1
                ibmax = imp_ll_max = kp1
                gomax = ig1
                iomax = imp_bs[kp2].val
                continue

            ig2 = gen_bs[kg2].val
            ip1 = d_implen - imp_bs[kp2].val

            # Check if impostor score range lies higher than genuine score range
            if impostors[ip1] > genuines[ig2]:
                gbmin = kg1
                ibmin = kp1
                gomin = ig2
                iomin = imp_bs[kp1].val
                gen_ll_min = kg2
                imp_ll_min = kp2
                if kg1 < head_gen_ll:
                    head_gen_ll = kg1
                if kp1 < head_imp_ll:
                    head_imp_ll = kp1
                centre_offset = 1
                continue
            # Some overlap has been detected, so we break.
            break
        sg1, sg2, sp1, sp2 = genuines[ig1], genuines[ig2], impostors[ip1], impostors[ip2]

        # Find FRR and FARs closest to the EER line.
        if sg1 >= sp1:
            rmin = head_gen_ll
            rmax = gen_bs[rmin].next
            imin = gen_bs[rmin].val

            # Minimize the range across which to search for the score
            while rmax != -1:
                imax = gen_bs[rmax].val
                if genuines[imin] < sp1 < genuines[imax]:
                    break
                rmin = rmax
                rmax = gen_bs[rmin].next
                imin = gen_bs[rmin].val
            # Use binary search to find the actual point
            # While drawing new values and keeping the list consistent
            if rmax == -1:
                rmax = d_genlen
                imax = d_genlen
            imid = (rmin + rmax) // 2
            while rmax - rmin > 1:
                rmid = (rmin + rmax) // 2
                imid = round(beta(rmid - rmin+1,rmax - rmid +1) * (imax - imin) + imin)

                vmid = genuines[imid]
                gen_bs[rmid].next = rmax if gen_bs[rmax].round == i_m else -1
                gen_bs[rmid].val = imid
                gen_bs[rmin].next = rmid
                if vmid == sp1:
                    break
                    pass
                elif vmid > sp1:
                    rmax = rmid
                    imax = imid
                elif vmid < sp1:
                    rmin = rmid
                    imin = imid
            ig1_ = imid

            rmin = head_gen_ll
            rmax = gen_bs[rmin].next
            imin = gen_bs[rmin].val
            # Minimize the range across which to search for the score
            while rmax != -1:
                imax = gen_bs[rmax].val
                if genuines[imin] < sp2 < genuines[imax]:
                    break
                rmin = rmax
                rmax = gen_bs[rmin].next
                imin = gen_bs[rmin].val
            # Use binary search to find the actual point
            # While drawing new values and keeping the list consistent
            if rmax == -1:
                rmax = d_genlen
                imax = d_genlen
            imid = (rmin + rmax) // 2
            while rmax - rmin > 1:
                rmid = (rmin + rmax) // 2
                imid = round(beta(rmid - rmin+1,rmax - rmid +1) * (imax - imin) + imin)
                vmid = genuines[imid]
                gen_bs[rmid].next = rmax if gen_bs[rmax].round == i_m else -1
                gen_bs[rmid].val = imid
                gen_bs[rmin].next = rmid
                if vmid == sp2:
                    break
                    pass
                elif vmid > sp2:
                    rmax = rmid
                    imax = imid
                elif vmid < sp2:
                    rmin = rmid
                    imin = imid
            ig2_ = imid
            frr_1 = <float>ig1_ / <float>d_genlen
            far_1 = 1. - (<float>ip1 / <float>d_implen)
            frr_2 = <float>ig2_ / <float>d_genlen
        else:
            rmin = head_imp_ll
            rmax = imp_bs[rmin].next
            imin = imp_bs[rmin].val
            # Minimize the range across which to search for the score
            while rmax != -1:
                imax = imp_bs[rmax].val
                if impostors[d_implen - imin] > sg1 > impostors[d_implen - imax]:
                    break
                rmin = rmax
                rmax = imp_bs[rmin].next
                imin = imp_bs[rmin].val
            # Use binary search to find the actual point
            # While drawing new values and keeping the list consistent
            if rmax == -1:
                rmax = d_implen
                imax = d_implen
            imid = (rmin + rmax) // 2
            while rmax - rmin > 1:
                rmid = (rmin + rmax) // 2
                imid = round(beta(rmid - rmin +1,rmax - rmid +1) * (imax - imin) + imin)
                vmid = impostors[d_implen - imid]
                imp_bs[rmid].next = rmax if imp_bs[rmin].round == i_m else -1
                imp_bs[rmid].val = imid
                imp_bs[rmin].next = rmid
                if vmid == sg1:
                    break
                    pass
                elif vmid < sg1:
                    rmax = rmid
                    imax = imid
                elif vmid > sg1:
                    rmin = rmid
                    imin = imid
            ip1_ = imid
            frr_1 = <float>ig1 / <float>d_genlen
            far_1 = <float>ip1_ / <float>d_implen
            frr_2 = <float>ig2 / <float>d_genlen

        if far_1 - frr_2 == 0:
            eers[i_m] = frr_2
        elif (far_1 - frr_1) / (far_1 - frr_2) <= 0.:
            eers[i_m] = far_1
        else:
            eers[i_m] = frr_2
    # Do a final sort of the array and determine the requested lower & upper bounds
    np.asarray(eers).sort(kind='quicksort')
    cdef int i_ci_lower = <int> m * ((1- ci)/2)
    cdef int i_ci_upper = <int> m * ((1+ci)/2)

    return eer,eers, eers[i_ci_lower],eers[i_ci_upper]

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef feer(np.ndarray[np.float64_t,ndim=1] impostors,np.ndarray[np.float64_t,ndim=1] genuines, is_sorted=False):
    """
    O(log n) function for calculating EERs. Consists of two phases.
    1. Determine the general location of the intersection between the EER line and the ROC curve.
    2. Expand the found point on the genuine line within the bound of the impostor scores (or vice-versa), to find the _actual_ intersection with the EER line.
    :param impostors: list of impostor scores 
    :param genuines: list of genuine scores
    :param is_sorted: whether lists are sorted or not (will sort if not)
    :return: EER
    """

    """Initialize required params"""
    cdef int implen, genlen, ig1_,ig2_,ip1_,ip2_, ig1,ip2,ig2,ip1
    cdef float sg1,sg2,sp1,sp2,pos, dep
    cdef float far_1,frr_1,frr_2,
    cdef float d_implen,d_genlen, imp_pos, gen_pos
    cdef int i
    """Sort both impostor and genuine lists in ascending order, if not already done"""
    if not is_sorted:
        impostors[::-1].sort(axis=0)
        genuines[::-1].sort(axis=0)

    """Get """
    genlen = genuines.shape[0]
    implen = impostors.shape[0]
    d_genlen = genlen - 1.
    d_implen = implen - 1.
    pos = .5
    dep = 1.
    if impostors[implen - 1] < genuines[0]:
        return 0.0
    elif genuines[genlen - 1] < impostors[0]:
        return 1.0
    for i in range(2*int(max(math.log2(genlen),math.log2(implen)))):
        ig1 = <int>((1. - pos) * d_genlen)
        ip2 = <int>ceil(pos * d_implen)
        if genuines[ig1] > impostors[ip2]:
            dep *= 2
            pos += 1. / dep
            continue
        ig2 = <int>ceil((1. - pos) * d_genlen)
        ip1 = <int>(pos * d_implen)
        if impostors[ip1] > genuines[ig2]:
            dep *= 2
            pos -= 1. / dep
            continue

        break
    sg1, sg2, sp1, sp2 = genuines[ig1], genuines[ig2], impostors[ip1], impostors[ip2]
    if sg1 >=  sp1:
        # Expand genuine scores within the bounds of impostor scores
        ig1 = bisect_right(genuines, sp1)
        ig2 = bisect_left(genuines, sp2) - 1
    else:
        # Expand the impostor scores within the bounds of genuine scores
        ip1 = bisect_right(impostors, sg1)
        ip2 = bisect_left(impostors, sg2)

    frr_1 = (ig1 / d_genlen)
    far_1 = 1. - (ip1 / d_implen)
    frr_2 = ig2 / d_genlen
    far_2 = 1. - (ip2 / d_implen)

    # Find intersection with EER line
    if far_1 - frr_2 == 0:
        return frr_2
    elif (far_1 - frr_1) / (far_1 - frr_2) <= 0.:
        return far_1
    else:
        return frr_2

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef bootstrap_draw_sorted(np.ndarray[np.float64_t,ndim=1] a,int samples=-1):
    cdef int i, j, d, r, c
    c = 0
    if samples == -1:
        samples = len(a)
    cdef int[:] counts = array((samples,),itemsize=sizeof(int),format='i')
    for i in range(samples):
        counts[i] = 0
    cdef np.ndarray[np.float64_t,ndim=1] out = np.zeros(samples)
    ran = random.randint
    cdef np.ndarray[np.long_t,ndim=1] rands = np.random.randint(0,samples,samples)
    for i in range(samples):
        counts[rands[i]] += 1
    for i in range(samples):
        d = counts[i]
        for j in range(d):
            out[c] = a[i]
            c += 1
    return out




