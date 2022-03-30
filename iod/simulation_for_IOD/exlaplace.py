# coding=utf-8
# From ICRS to ITRS.
import numpy as np
import math
import random
import scipy
from datetime import datetime,timedelta
from astropy.time import Time
from astropy.utils import iers

iers.conf.iers_auto_url = 'ftp://cddis.gsfc.nasa.gov/pub/products/iers/finals2000A.all'
iers_b = iers.IERS_B.open()
# 千米、千克、秒、弧度。
GM = 398600.4415  # JGM-3
R = 6378.137  # WGS-84
f = 1 / 298.257223563  # WGS-84
mu = 398600.4415

I = P = np.array([1, 0, 0])
J = Q = np.array([0, 1, 0])
K = W = np.array([0, 0, 1])


class Satellite:
    # to initialize the location by r,velocity by v,and so on.
    def __init__(self, r='', v='', a='', e='', i='', Omega='', omega='', nu=''):
        if any(r):
            self.r = np.array(r)
            self.v = np.array(v)
            self.h = np.cross(self.r, self.v)
            self.n = np.cross(K, self.h)
        if any(a):
            self.a = a
            self.e = e
            self.i = i
            self.Omega = Omega
            self.omega = omega
            self.nu = nu
            self.R = np.zeros((3, 3))

        # to calculate the transformation matrice.

    def cal_trans_matrice(self):
        self.R[0][0] = math.cos(self.Omega) * math.cos(self.omega) - math.sin(self.Omega) * math.sin(
            self.omega) * math.cos(self.i)
        self.R[0][1] = -math.cos(self.Omega) * math.sin(self.omega) - math.sin(self.Omega) * math.cos(
            self.omega) * math.cos(self.i)
        self.R[0][2] = math.sin(self.Omega) * math.sin(self.i)
        self.R[1][0] = math.sin(self.Omega) * math.cos(self.omega) + math.cos(self.Omega) * math.sin(
            self.omega) * math.cos(self.i)
        self.R[1][1] = -math.sin(self.Omega) * math.sin(self.omega) + math.cos(self.Omega) * math.cos(
            self.omega) * math.cos(self.i)
        self.R[1][2] = -math.cos(self.Omega) * math.sin(self.i)
        self.R[2][0] = math.sin(self.omega) * math.sin(self.i)
        self.R[2][1] = math.cos(self.omega) * math.sin(self.i)
        self.R[2][2] = math.cos(self.i)

    # to calculate r and v from elements.
    def from_elements_to_rv(self):
        # to calculate r and v in the Perifocal frame.
        self.p = self.a * (1 - self.e ** 2)
        self.r_magnitude = self.p / (1 + self.e * math.cos(self.nu))
        self.r_per = self.r_magnitude * math.cos(self.nu) * P + self.r_magnitude * math.sin(self.nu) * Q
        self.v_per = ((mu / self.p) ** 0.5) * (-math.sin(self.nu) * P + (self.e + math.cos(self.nu)) * Q)
        # to transform the r and v from Perifocal frame to Geocentric-Equatorial frame.
        self.cal_trans_matrice()
        self.r = np.dot(self.R, self.r_per)
        self.v = np.dot(self.R, self.v_per)

    def from_rv_to_elements(self):
        try:
            self.e_vector = ((np.dot(self.v, self.v) - (mu / np.linalg.norm(self.r))) * self.r - (
                        np.dot(self.r, self.v) * self.v)) / mu
            self.e = np.linalg.norm(self.e_vector)
            self.a = np.dot(self.h, self.h) / (mu * (1 - self.e ** 2))
            self.i = math.acos(self.h[2] / np.linalg.norm(self.h))
            self.Omega = math.acos(self.n[0] / np.linalg.norm(self.n))
            if self.n[1] < 0:
                self.Omega = self.Omega + 3.14
            self.omega = math.acos(np.dot(self.n, self.e_vector) / (np.linalg.norm(self.n) * self.e))
            if self.e_vector[2] < 0:
                self.omega = self.omega + 3.14
            self.nu = math.acos(np.dot(self.e_vector, self.r) / (self.e * np.linalg.norm(self.r)))
            if np.dot(self.r, self.v) < 0:
                self.nu = self.nu + 3.14
            self.E = math.acos((self.e + math.cos(self.nu)) / (1 + self.e * math.cos(self.nu)))
            self.M = self.E - self.e * math.sin(self.E)
        except:
            pass


class Coordinate(object):
    def __init__(self, t):
        self.UTC = Time(t, scale='utc')  # Import UTC time.
        # WGS84参考椭球.

    def TT(self):
        TT = self.UTC.tt  # Transform UTC to TT.
        return TT

    def UT1(self):
        UT1 = self.UTC.ut1  # Transform UTC to UT1.
        return UT1

    def JD(self, T):
        JD = T.jd  # Transform TT to JD of TT.
        return JD

    def second_of_day(self, T):
        Temp = str(T).split('T')[1]
        T_str = Temp.split(':')
        # Get the second of the day in T.
        Second = float(T_str[0]) * 3600 + float(T_str[1]) * 60 + float(T_str[2])
        return Second

    # 岁差矩阵，历元T时刻的平赤道面和平春分点相对于J2000平赤道和平春分点，T为地球时从J2000TT时刻起算的儒略世纪数。
    def Precession(self):
        TT = self.TT()
        JD_TT = self.JD(TT)
        # 输入时历元T时刻TT的儒略日数。
        T = (JD_TT - 2451545.0) / 36525.0  # 历元T时刻从J2000TT时刻起的儒略世纪数。
        zeta = math.radians((2306.2182 * T + 0.30188 * (T ** 2) + 0.017998 * (T ** 3)) / 3600)
        theta = math.radians((2004.3109 * T - 0.42665 * (T ** 2) - 0.041833 * (T ** 3)) / 3600)
        z = math.radians((2306.2182 * T + 0.30188 * (T ** 2) + 0.017998 * (T ** 3) + 0.79280 * (T ** 2) + 0.000205 * (
                    T ** 3)) / 3600)
        P = np.zeros((3, 3))
        P[0][0] = -math.sin(z) * math.sin(zeta) + math.cos(z) * math.cos(theta) * math.cos(zeta)
        P[1][0] = math.cos(z) * math.sin(zeta) + math.sin(z) * math.cos(theta) * math.cos(zeta)
        P[2][0] = math.sin(theta) * math.cos(zeta)
        P[0][1] = -math.sin(z) * math.cos(zeta) - math.cos(z) * math.cos(theta) * math.sin(zeta)
        P[1][1] = math.cos(z) * math.cos(zeta) - math.sin(z) * math.cos(theta) * math.sin(zeta)
        P[2][1] = -math.sin(theta) * math.sin(zeta)
        P[0][2] = -math.cos(z) * math.sin(theta)
        P[1][2] = -math.sin(z) * math.sin(theta)
        P[2][2] = math.cos(theta)
        return P

    # IAU1980框架下的章动矩阵，瞬时平坐标系（平赤道和平春分点）到瞬时真坐标系（真赤道和真春分点）。
    def Nutation(self):
        TT = self.TT()
        JD_TT = self.JD(TT)
        T = (JD_TT - 2451545.0) / 36525.0
        DPsi = 0  # 春分点周期性变化。
        DEpsilon = 0  # 黄赤交角周期性变化。
        Epsilon = math.radians(
            (23.43929111 * 3600 - 46.8150 * T - 0.00059 * (T ** 2) + 0.001813 * (T ** 3)) / 3600)  # 平黄赤交角。
        # IAU1980章动序列。
        pl = [0, 0, -2, 2, -2, 1, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, -2]
        pl_i = [0, 0, 0, 0, 0, -1, -2, 0, 0, 1, 1, -1, 0, 0, 0, 2, 1, 2, -1, 0]
        pF = [0, 0, 2, -2, 2, 0, 2, -2, 2, 0, 2, 2, 2, 0, 2, 0, 0, 2, 0, 0]
        pD = [0, 0, 0, 0, 0, -1, -2, 0, -2, 0, -2, -2, -2, -2, -2, 0, 0, -2, 0, 2]
        p_OMEGA = [1, 2, 1, 0, 2, 0, 1, 1, 2, 0, 2, 2, 1, 0, 0, 0, 1, 2, 1, 1]
        dPsi = [-171996 - 174.2 * T, 2062 + 0.2 * T, 46, 11, -3, -3, -2, 1, -13187 - 1.6 * T, 1426 - 3.4 * T,
                -517 + 1.2 * T, 217 - 0.5 * T, 129 + 0.1 * T, 48, -22, 17 - 0.1 * T, -15, -16 + 0.1 * T, -12, -6]
        dEpsilon = [92025 + 8.9 * T, -895 + 0.5 * T, -24, 0, 1, 0, 1, 0, 5736 - 3.1 * T, 54 - 0.1 * T, 224 - 0.6 * T,
                    -95 + 0.3 * T, -70, 1, 0, 0, 9, 7, 6, 3]
        l = (134 * 3600 + 57 * 60 + 46.733) + (477198 * 3600 + 52 * 60 + 2.633) * T + 31.310 * (T ** 2) + 0.064 * (
                    T ** 3)  # 月球平近点角。
        l_i = (357 * 3600 + 31 * 60 + 39.804) + (35999 * 3600 + 3 * 60 + 1.224) * T - 0.577 * (T ** 2) - 0.012 * (
                    T ** 3)  # 太阳平近点角。
        F = (93 * 3600 + 16 * 60 + 18.877) + (483202 * 3600 + 1 * 60 + 3.137) * T - 13.257 * (T ** 2) + 0.011 * (
                    T ** 3)  # 月球升交距角。
        D = (297 * 3600 + 51 * 60 + 1.307) + (445267 * 3600 + 6 * 60 + 41.328) * T - 6.891 * (T ** 2) + 0.019 * (
                    T ** 3)  # 日月间平经度差。
        OMEGA = (125 * 3600 + 2 * 60 + 40.280) - (1934 * 3600 + 8 * 60 + 10.539) * T + 7.455 * (T ** 2) + 0.008 * (
                    T ** 3)  # 月球轨道升交点平经度。
        for i in range(20):
            phi = math.radians((pl[i] * l + pl_i[i] * l_i + pF[i] * F + pD[i] * D + p_OMEGA[i] * OMEGA) / 3600)
            DPsi = math.radians((math.degrees(DPsi) * 3600 + dPsi[i] * math.sin(phi) * 0.0001) / 3600)
            DEpsilon = math.radians((math.degrees(DEpsilon) * 3600 + dEpsilon[i] * math.cos(phi) * 0.0001) / 3600)
        Epsilon_i = Epsilon + DEpsilon
        N = np.zeros((3, 3))
        N[0][0] = math.cos(DPsi)
        N[1][0] = math.cos(Epsilon_i) * math.sin(DPsi)
        N[2][0] = math.sin(Epsilon_i) * math.sin(DPsi)
        N[0][1] = -math.cos(Epsilon) * math.sin(DPsi)
        N[1][1] = math.cos(Epsilon) * math.cos(Epsilon_i) * math.cos(DPsi) + math.sin(Epsilon) * math.sin(Epsilon_i)
        N[2][1] = math.cos(Epsilon) * math.sin(Epsilon_i) * math.cos(DPsi) - math.sin(Epsilon) * math.cos(Epsilon_i)
        N[0][2] = -math.sin(Epsilon) * math.sin(DPsi)
        N[1][2] = math.sin(Epsilon) * math.cos(Epsilon_i) * math.cos(DPsi) - math.cos(Epsilon) * math.sin(Epsilon_i)
        N[2][2] = math.sin(Epsilon) * math.sin(Epsilon_i) * math.cos(DPsi) + math.cos(Epsilon) * math.cos(Epsilon_i)
        return N, DPsi, Epsilon

    # 瞬时真赤道坐标系和准地固系之间的旋转矩阵.
    def Rotation(self):
        UT1 = self.UT1()
        JD_UT1 = self.JD(UT1)
        UT1_second = self.second_of_day(UT1)
        DPsi = self.Nutation()[1]
        Epsilon = self.Nutation()[2]
        T = (JD_UT1 - 2451545) / 36525
        if JD_UT1 - int(JD_UT1) > 0.5:
            T0 = (int(JD_UT1) - 2451545 + 0.5) / 36525
        else:
            T0 = (int(JD_UT1) - 2451545 - 0.5) / 36525
        GMST = ((24110.54841 + 8640184.812866 * T0 + 1.002737909350795 * UT1_second + 0.093104 * (
                    T ** 2) - 0.0000062 * (T ** 3)) * np.pi) / (3600 * 12)
        GAST = GMST + DPsi * math.cos(Epsilon)
        R = np.zeros((3, 3))
        R[0][0] = math.cos(GAST)
        R[0][1] = math.sin(GAST)
        R[1][0] = -math.sin(GAST)
        R[1][1] = math.cos(GAST)
        R[2][2] = 1
        return R

    # 极移矩阵.
    def Polar_Wandering(self):
        UTC = self.UTC
        Pm_x = float(str(iers_b.pm_xy(UTC)[0]).split(' ')[0])
        Pm_y = float(str(iers_b.pm_xy(UTC)[1]).split(' ')[0])
        x = math.radians(Pm_x / 3600)
        y = math.radians(Pm_y / 3600)
        Rx = np.zeros((3, 3))
        Ry = np.zeros((3, 3))
        Rx[0][0] = 1
        Rx[1][1] = math.cos(-y)
        Rx[1][2] = math.sin(-y)
        Rx[2][1] = -math.sin(-y)
        Rx[2][2] = math.cos(-y)
        Ry[0][0] = math.cos(-x)
        Ry[0][2] = -math.sin(-x)
        Ry[1][1] = 1
        Ry[2][0] = math.sin(-x)
        Ry[2][2] = math.cos(-x)
        PW = np.dot(Ry, Rx)
        return PW

    def ICRS_to_ITRS(self, x, y, z):
        # 输入ICRS直角坐标，输出ITRS直角坐标。
        r_icrs = [x, y, z]
        P = self.Precession()
        N = self.Nutation()[0]
        R = self.Rotation()
        PW = self.Polar_Wandering()
        U = np.dot(np.dot(np.dot(PW, R), N), P)
        r_itrs = np.dot(U, r_icrs)
        return r_itrs

    def ITRS_to_ICRS(self, x, y, z):
        # 输入ITRS直角坐标，输出ICRS直角坐标。
        r_itrs = [x, y, z]
        P = self.Precession()
        N = self.Nutation()[0]
        R = self.Rotation()
        PW = self.Polar_Wandering()
        U = np.dot(np.dot(np.dot(PW, R), N), P)
        V = np.linalg.inv(U)
        r_icrs = np.dot(V, r_itrs)
        return r_icrs

    def Geodetic_to_ICRS(self, lon, lat, h):
        # 输入大地经纬度，海平面高度和时间，输出ICRS.
        N = R / math.sqrt(1 - f * (2 - f) * (math.sin(lat) ** 2))
        r_itrs = [(N + h) * math.cos(lat) * math.cos(lon), (N + h) * math.cos(lat) * math.sin(lon),
                  (((1 - f) ** 2) * N + h) * math.sin(lat)]
        r_icrs = self.ITRS_to_ICRS(r_itrs[0], r_itrs[1], r_itrs[2])
        return r_icrs

    def Geodetic_to_ITRS(self, lon, lat, h):
        # 输入大地经纬度，海平面高度和时间，输出ITRS.
        N = R / math.sqrt(1 - f * (2 - f) * (math.sin(lat) ** 2))
        r_itrs = [(N + h) * math.cos(lat) * math.cos(lon), (N + h) * math.cos(lat) * math.sin(lon),
                  (((1 - f) ** 2) * N + h) * math.sin(lat)]
        return r_itrs

    def Topocentric_to_ICRS(self, Az, El, range, x='', y='', z='', lon='', lat='', h=''):
        # 输入站心坐标系下的角度与距离值、观测站点的位置坐标.
        # 如果站点输入的是ITRS坐标.
        if x:
            # 站心坐标系中，目标与地心连线矢量的表示.
            rho_e = range * math.cos(El) * math.sin(Az)
            rho_n = range * math.cos(El) * math.cos(Az)
            rho_z = range * math.sin(El)
            r = [rho_e, rho_n, rho_z + np.sqrt(x ** 2 + y ** 2 + z ** 2)]
            # 目标从站心坐标系到ITRS.
            site = [x, y, z]
            lon = np.arccos(x / math.sqrt(x ** 2 + y ** 2))  # 测站在xoy平面投影向量与ITRS的X轴夹角.
            lat = np.pi / 2 - np.arccos(z / np.linalg.norm(site))  # 测站与ITRS的Z轴夹角.
            S = np.zeros((3, 3))
            S[0][0] = -math.sin(lon)
            S[0][1] = -math.sin(lat) * math.cos(lon)
            S[0][2] = math.cos(lon) * math.cos(lat)
            S[1][0] = math.cos(lon)
            S[1][1] = -math.sin(lon) * math.sin(lat)
            S[1][2] = math.sin(lon) * math.cos(lat)
            S[2][0] = 0
            S[2][1] = math.cos(lat)
            S[2][2] = math.sin(lat)
            r_itrs = np.dot(S, r)
            r_icrs = self.ITRS_to_ICRS(r_itrs[0], r_itrs[1], r_itrs[2])
            return r_icrs


def Data(file):
    R = []  # 测站坐标(ITRS).
    L = []  # 不同时刻观测到的卫星单位方向向量(ICRS).
    T = []  # 不同时刻.
    T_origin = []
    with open(file) as f:
        data = f.readlines()
        i = 0
        for line in data:
            a = line.split()
            a = list(map(float, a))
            if i != 0:
                T_origin.append(a[0:6])
                ra = np.radians(a[6])
                dec = np.radians(a[7])

                l = [math.cos(dec) * math.cos(ra), \
                     math.cos(dec) * math.sin(ra), math.sin(dec)]
                L.append(l)
            elif i == 0:
                R = a
            i = i + 1
        # 改变时间为标准格式.
        for t in T_origin:
            t = str(int(t[0])) + '-' + str(int(t[1])) + '-' + \
                str(int(t[2])) + 'T' + str(int(t[3])) + ':' + \
                str(int(t[4])) + ':' + str(t[5])
            T.append(t)
    return [R, L, T]


def ExLaplace(R, L, T):
    mu = 1  # mu=GM=1.
    a_e = 6378.137  # 长度单位量纲为地球半径.
    T_u = math.sqrt(a_e ** 3 / 398600.4415)  # 时间单位量纲.
    site = [R[0] * 0.001, R[1] * 0.001, R[2] * 0.001]  # 观测站点在ICRS中的坐标.
    t0 = Time(T[0], scale='utc')  # 取t0时刻.
    T.pop(0)  # 把t0时刻及其观测数据剔除.
    L.pop(0)
    F = []
    G = []
    for i in range(1000):  # 迭代f和g.
        print(i)
        j = 0
        for t, l in zip(T, L):
            t = Time(t, scale='utc')  # 取每一次观测时间.
            tau = ((t - t0).sec) / T_u  # 将时间差在指定量纲下单位化.
            if i == 0:  # 若f和g是第一次迭代.
                f = 1
                g = tau
            else:  # 若不是第一次迭代.
                f = 1 - (1 / 2) * u[3] * (tau ** 2) + (1 / 2) * u[5] * sigma * (tau ** 3) + \
                    (1 / 24) * u[5] * (3 * V - 2 * u[1] - 15 * u[2] * (sigma ** 2)) * (tau ** 4) + \
                    (1 / 8) * u[7] * sigma * (-3 * V + 2 * u[1] + 7 * u[2] * sigma ** 2) * (tau ** 5) + \
                    (1 / 720) * u[7] * (u[2] * (sigma ** 2) * (630 * V - 420 * u[1] - 945 * u[2] * (sigma ** 2)) \
                                        - (22 * u[2] - 66 * u[1] * V + 45 * (V ** 2))) * (tau ** 6)
                g = tau - (1 / 6) * u[3] * (tau ** 3) + (1 / 4) * u[5] * sigma * (tau ** 4) + (1 / 120) * u[5] * \
                    (9 * V - 8 * u[1] - 45 * u[2] * (sigma ** 2)) * (tau ** 5) + \
                    (1 / 24) * u[7] * sigma * (-6 * V + 5 * u[1] + 14 * u[2] * (sigma ** 2)) * (tau ** 6)
            # 通过每一次观测角度，计算每一次的叉乘矩阵.
            A = np.array([[0, -l[2], l[1]], [l[2], 0, -l[0]], [-l[1], l[0], 0]])
            # 通过每一次观测时间，计算测站的ICRS坐标.
            # print(t)
            # print(site[0], site[1], site[2])
            site_t = Coordinate(t)
            R = site_t.ITRS_to_ICRS(site[0], site[1], site[2])
            # print(R)
            # 将测站坐标在指定单位量纲下单位化.
            R = [R[0] / a_e, R[1] / a_e, R[2] / a_e]
            d = np.dot(A, R)
            c = np.append(np.dot(f, A), np.dot(g, A), axis=1)
            if j != 0:
                C = np.append(C, c, axis=0)
                D = np.append(D, d, axis=0)
            elif j == 0:
                C = c
                D = d
            j = j + 1
        # x = np.dot(np.linalg.inv(np.dot(np.transpose(C),C)),np.dot(np.transpose(C),D))
        x = np.linalg.lstsq(C, D, rcond=-1)[0]
        r = np.array(x[0:3])
        v = np.array(x[3:6])
        sigma = np.dot(r, v)
        V = np.dot(v, v)
        u = [mu / (np.linalg.norm(r) ** 0), mu / (np.linalg.norm(r) ** 1), mu / (np.linalg.norm(r) ** 2), \
             mu / (np.linalg.norm(r) ** 3), mu / (np.linalg.norm(r) ** 4), mu / (np.linalg.norm(r) ** 5), \
             mu / (np.linalg.norm(r) ** 6), mu / (np.linalg.norm(r) ** 7), mu / (np.linalg.norm(r) ** 8)]
        r_0 = r * a_e
        v_0 = v * a_e / T_u
        F.append(f)
        G.append(g)
        try:
            if (i > 1) and (abs(F[i] - F[i - 1]) < 1E-11) and (abs(G[i] - G[i - 1]) < 1E-11):
                satellite = Satellite(r=r_0, v=v_0)
                satellite.from_rv_to_elements()
                return [satellite.a, satellite.e, math.degrees(satellite.i), \
                        math.degrees(satellite.Omega), math.degrees(satellite.omega), \
                        math.degrees(satellite.M)]
                break
            elif i == 999:
                satellite = Satellite(r=r_0, v=v_0)
                satellite.from_rv_to_elements()
                return [satellite.a, satellite.e, math.degrees(satellite.i), \
                        math.degrees(satellite.Omega), math.degrees(satellite.omega), \
                        math.degrees(satellite.M)]
                break
        except:
            continue


obs1 = Data('./Week2/obs1.dat')
obs1 = Data('numbers.txt')
elements = ExLaplace(obs1[0], obs1[1], obs1[2])
print(elements)



