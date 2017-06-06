# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

samples = np.array([[3.5010000000000000e+004, - 4.9388819984325000e+003, - 2.1388001992055579e+003],
                    [3.5040000000000000e+004, - 4.9323739883349981e+003, - 2.1468546345586074e+003],
                    [3.5070000000000000e+004, - 4.9192534204187332e+003, - 2.1465596313587926e+003],
                    [3.5100000000000000e+004, - 4.9450461102077907e+003, - 2.1450393112224374e+003],
                    [3.5130000000000000e+004, - 4.9367847926433160e+003, - 2.1554265554534795e+003],
                    [3.5160000000000000e+004, - 4.9352710685415514e+003, - 2.1496889403144532e+003],
                    [3.5190000000000000e+004, - 4.9254990043418729e+003, - 2.1485146088590368e+003],
                    [3.5220000000000000e+004, - 4.9238668528838189e+003, - 2.1478122457006407e+003],
                    [3.5250000000000000e+004, - 4.9371960864495986e+003, - 2.1549087106247321e+003],
                    [3.5280000000000000e+004, - 4.9506156372971936e+003, - 2.1472000030987124e+003]])

# samples = np.array([[1.8240000000000000e+004, -2.0033220728151043e+003, -6.3352245411248578e+002],
#                     [1.8270000000000000e+004, -1.9893229902751784e+003, -5.2253930530827301e+002],
#                     [1.8300000000000000e+004, -1.8857416390313615e+003, -4.5877597841280846e+002],
#                     [1.8330000000000000e+004, -1.8824594492544195e+003, -4.5778550366230809e+002],
#                     [1.8360000000000000e+004, -1.8787030972648129e+003, -4.5577420183014857e+002],
#                     [1.8390000000000000e+004, -1.8749467452752060e+003, -4.5376289999798917e+002],
#                     [1.8420000000000000e+004, -1.7779117543682057e+003, -3.9492044147231024e+002],
#                     [1.8450000000000000e+004, -1.6981320355432504e+003, -3.2189642261794791e+002],
#                     [1.8480000000000000e+004, -1.6985083566258788e+003, -3.2133983598157511e+002],
#                     [1.8510000000000000e+004, -1.6986638996610081e+003, -3.1600899260109350e+002]])


imissing = np.random.random_integers(0, 9)
missing = samples[imissing, :]
samples = np.delete(samples, imissing, axis=0)
missing_t = missing[0]
missing_x = missing[1]
missing_y = missing[2]

t = samples[:, 0]
x = samples[:, 1]
y = samples[:, 2]


def lagrange(xs, ys, xnew):
    ans = 0.0

    # 两重循环实现插值
    for i in range(len(ys)):
        temp = ys[i]
        for j in range(len(ys)):
            if i != j:
                temp *= (xnew - xs[j]) / (xs[i] - xs[j])
        ans += temp
    return ans


def spline(xs, ys, xnew):

    tck = interpolate.splrep(xs, ys)
    ans = interpolate.splev(xnew, tck, der=0)
    return ans


import numpy

slot_trace = 30
length_sequence_input = 10
length_sequence_output = 10


def update(trace, samples):
    modified = False
    for sample in samples:
        t = sample[0]

        # delete old one
        entry = numpy.where(trace[:, 0] == t)
        if numpy.size(entry) != 0:
            old = trace[entry[0]]
            if old != sample:
                modified = True
            trace = numpy.delete(trace, entry, axis=0)

        entry = numpy.searchsorted(trace[:, 0], t, side='left')
        trace = numpy.insert(trace, entry, sample, axis=0)

    return trace, modified


trace_ = numpy.array([[0, 100, -100],
                      [30, 100, -100],
                      [60, 100, -100],
                      [90, 100, -100],
                      [120, 100, -100],
                      [150, 100, -100],
                      [180, 100, -100]])

samples_ = numpy.array([[110, 100, -100],
                        [-10, 100, -100],
                        [60, 200, -100],
                        [200, 100, -100],
                        [10, 200, -100]])

trace_, modified = update(trace_, samples_)


def select_samples(trace, instant_target):
    len_full = length_sequence_input
    t_end_min = instant_target - length_sequence_output * slot_trace
    t_end = numpy.maximum(trace[:, 0])
    entry_end = numpy.searchsorted(trace[:, 0], t_end, side='right')
    if t_end < t_end_min:
        len_short = (t_end_min - t_end) / slot_trace
        len_full -= len_short
    t_start = t_end - (len_full - 1) * slot_trace
    entry_start = numpy.searchsorted(trace[:, 0], t_start)
    if numpy.where(trace[:, 0] == t_start).size == 0:
        entry_start -= 1
    samples = trace[entry_start:entry_end, :]


def recover_trace(samples, instants):
    interpolate = spline

    for instant in instants:
        ts = samples[:, 0]
        xs = samples[:, 1]
        ys = samples[:, 2]

        # xnew = Lagrange(t, x, tnew)
        # ynew = Lagrange(t, y, tnew)
        xnew = interpolate(ts, xs, instant)
        ynew = interpolate(ts, ys, instant)
        entry = numpy.searchsorted(ts, instant)
        samples = numpy.insert(samples, entry, [instant, xnew, ynew], axis=0)

    pass


# 开始插值
tnew = missing_t

# xnew = Lagrange(t, x, tnew)
# ynew = Lagrange(t, y, tnew)
xnew = spline(t, x, tnew)
ynew = spline(t, y, tnew)

# for i in range(len(xn)):
#     yn[i] = Lagrange(x, y, xn[i])


def show(m, n, mnew, nnew, description):
    plt.figure()
    plt.plot(m[:imissing], n[:imissing], 'b-o')  # 已知结点
    if imissing - 1 >= 0:
        plt.plot([m[imissing - 1], mnew], [n[imissing - 1], nnew], 'r--')  # 插值结果
    plt.plot(mnew, nnew, 'r*')  # 插值结果
    if imissing < 9:
        plt.plot([mnew, m[imissing]], [nnew, n[imissing]], 'g--')  # 插值结果
    plt.plot(m[imissing:], n[imissing:], 'b-o')  # 已知结点
    plt.title(description)
    plt.show()

show(t, x, tnew, xnew, 't-x')
show(t, y, tnew, ynew, 't-y')
show(x, y, xnew, ynew, 'x-y')
#
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import interpolate
#
# # Cubic-spline
#
# # x = np.arange(0,2*np.pi+np.pi/4,2*np.pi/8)
# # y = np.sin(x)
# # tck, _, _, _ = interpolate.splrep(x,y,s=0, full_output=True)
# # xnew = np.arange(0,2*np.pi,np.pi/50)
# # ynew = interpolate.splev(xnew,tck,der=0)
# samples = np.array([[3.5010000000000000e+004, - 4.9388819984325000e+003, - 2.1388001992055579e+003],
#                     [3.5040000000000000e+004, - 4.9323739883349981e+003, - 2.1468546345586074e+003],
#                     [3.5070000000000000e+004, - 4.9192534204187332e+003, - 2.1465596313587926e+003],
#                     [3.5100000000000000e+004, - 4.9450461102077907e+003, - 2.1450393112224374e+003],
#                     [3.5130000000000000e+004, - 4.9367847926433160e+003, - 2.1554265554534795e+003],
#                     [3.5160000000000000e+004, - 4.9352710685415514e+003, - 2.1496889403144532e+003],
#                     [3.5190000000000000e+004, - 4.9254990043418729e+003, - 2.1485146088590368e+003],
#                     [3.5220000000000000e+004, - 4.9238668528838189e+003, - 2.1478122457006407e+003],
#                     [3.5250000000000000e+004, - 4.9371960864495986e+003, - 2.1549087106247321e+003],
#                     [3.5280000000000000e+004, - 4.9506156372971936e+003, - 2.1472000030987124e+003]])
# irow_missing = np.random.random_integers(1, 8)
# missing = samples[irow_missing, :]
# samples = np.delete(samples, irow_missing, axis=0)
# missing_x = missing[1]
# missing_y = missing[2]
#
# x = samples[:, 1]
# y = samples[:, 2]
# tck, _, _, _ = interpolate.splrep(x,y,s=0, full_output=True)
# xnew = (samples[irow_missing, 1] + samples[irow_missing + 1, 1]) / 2
# xnew = np.array([xnew])
# ynew = interpolate.splev(xnew,tck,der=0)
#
# # plt.figure()
# # plt.plot(x,y,'x',xnew,ynew,xnew,np.sin(xnew),x,y,'b')
# # plt.legend(['Linear','Cubic Spline', 'True'])
# # plt.axis([-0.05,6.33,-1.05,1.05])
# # plt.title('Cubic-spline interpolation')
# # plt.show()
#
# plt.figure()
# plt.plot(x,y,'x',xnew,ynew,missing_x,missing_y,x,y,'b')
# plt.legend(['Linear','Cubic Spline', 'True'])
# # plt.axis([-0.05,6.33,-1.05,1.05])
# plt.title('Cubic-spline interpolation')
# plt.show()
#
# # Derivative of spline
#
# yder = interpolate.splev(xnew,tck,der=1)
# plt.figure()
# plt.plot(xnew,yder,xnew,np.cos(xnew),'--')
# plt.legend(['Cubic Spline', 'True'])
# plt.axis([-0.05,6.33,-1.05,1.05])
# plt.title('Derivative estimation from spline')
# plt.show()
#
# # Integral of spline
#
# def integ(x,tck,constant=-1):
#     x = np.atleast_1d(x)
#     out = np.zeros(x.shape, dtype=x.dtype)
#     for n in xrange(len(out)):
#         out[n] = interpolate.splint(0,x[n],tck)
#     out += constant
#     return out
# # >>>
# yint = integ(xnew,tck)
# plt.figure()
# plt.plot(xnew,yint,xnew,-np.cos(xnew),'--')
# plt.legend(['Cubic Spline', 'True'])
# plt.axis([-0.05,6.33,-1.05,1.05])
# plt.title('Integral estimation from spline')
# plt.show()
#
# # Roots of spline
#
# print interpolate.sproot(tck)
# # [ 0.      3.1416]
#
# # Parametric spline
#
# t = np.arange(0,1.1,.1)
# x = np.sin(2*np.pi*t)
# y = np.cos(2*np.pi*t)
# tck,u, _, _, _ = interpolate.splprep([x,y],s=0)
# unew = np.arange(0,1.01,0.01)
# out = interpolate.splev(unew,tck)
# plt.figure()
# plt.plot(x,y,'x',out[0],out[1],np.sin(2*np.pi*unew),np.cos(2*np.pi*unew),x,y,'b')
# plt.legend(['Linear','Cubic Spline', 'True'])
# plt.axis([-1.05,1.05,-1.05,1.05])
# plt.title('Spline of parametrically-defined curve')
# plt.show()
#
# # # Define function over sparse 20x20 grid
# # x,y = np.mgrid[-1:1:20j,-1:1:20j]
# # z = (x+y)*np.exp(-6.0*(x*x+y*y))
# #
# # plt.figure()
# # plt.pcolor(x,y,z)
# # plt.colorbar()
# # plt.title("Sparsely sampled function.")
# # plt.show()
# #
# # # Interpolate function over new 70x70 grid
# # xnew,ynew = np.mgrid[-1:1:70j,-1:1:70j]
# # tck = interpolate.bisplrep(x,y,z,s=0)
# # znew = interpolate.bisplev(xnew[:,0],ynew[0,:],tck)
# #
# # plt.figure()
# # plt.pcolor(xnew,ynew,znew)
# # plt.colorbar()
# # plt.title("Interpolated function.")
# # plt.show()