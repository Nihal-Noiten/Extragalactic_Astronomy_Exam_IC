import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.ticker import FormatStrFormatter, MultipleLocator, FuncFormatter
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib import rc
import time
import datetime
from scipy.integrate import quad, ode
# from scipy.special import erf
import argparse
import sys
from timeit import default_timer as timer
import os
from sys import stdout
from time import sleep
from astropy.modeling import models, fitting

############################################################################################################################

# LATEX: ON, MNRAS template

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",				# or sans-serif
    "font.serif": ["Times New Roman"]})	# or Helvetica

############################################################################################################################

if len(sys.argv) > 3:
	sys.exit('ARGV ERROR, TRY:   python   IC_Exam.py'+'\n'+'ARGV ERROR, TRY:   python   IC_Exam.py   save=Y/N'+'\n'+'ARGV ERROR, TRY:   python   IC_Exam.py   save=Y/N    N_particles')

if len(sys.argv) == 1:
	N = 10000
	save = 'save=N'
elif len(sys.argv) == 2:
	N = 10000
	save = sys.argv[1]
elif len(sys.argv) == 3:
	N = int(sys.argv[2])
	save = sys.argv[1]

print('\n'+'Building Initial Conditions:     N={:d}     {:}'.format(N, save))

# Set up to obtain the accurate conversion factors from internal units to physical units:

G_cgs = 6.67430e-8			# cm^3 g^-1 s^-2
pc_cgs = 3.08567758e18		# cm
Msun_cgs = 1.98855e33 		# g
Myr_cgs = 31557600. * 1e6 	# s

# Conversion factors from internal units to the chosen physical units:

G0 = 1.
R0 = 5.													# pc
M0 = 1e4												# Msun
V0 = np.sqrt( G_cgs * (Msun_cgs * M0) / (pc_cgs * R0) )	# cm/s
T0 = (pc_cgs * R0) / V0									# s
T0 = T0 / Myr_cgs										# Myr
V0 = V0 / 1e5 											# km/s

# Parameters in IU:	 # Giant Molecular Cloud?

# N     = 10000
M_tot = 1.		# M = 10^4 Msun
a     = 1.		# a = 5 pc
rho   = M_tot / ( 4. * np.pi / 3. * a**2)
t_dyn = np.sqrt( 3 * np.pi / ( 32. * rho) ) # t_IU
t_cgs = t_dyn * T0
soft  = 0.1 * a * np.cbrt(4. * np.pi / 3. / N) # r_IU

print()
print('Conversion factors to physical units:')
print('1 r_IU = {:.3f} pc'.format(R0))
print('1 m_IU = {:.0f} M_sun'.format(M0))
print('1 v_IU = {:.3f} km/s'.format(V0))
print('1 t_IU = {:.3f} Myr'.format(T0))
print()
print('Initial parameters in internal units and physical units:')
print("    N = {:d}".format(N))
print('    a = {:.4f} r_IU = {:.4f} pc'.format(a, a*R0))
print('  rho = {:.4f} d_IU = {:.3f} Msun pc^-3'.format(rho, rho * M0 / R0**3))
print('M_tot = {:.4f} m_IU = {:.0f}  Msun'.format(M_tot, M_tot*M0))
print('    m = {:.4f} m_IU = {:.4f} Msun'.format(M_tot/N, M_tot/N*M0))
print('t_dyn = {:.4f} t_IU = {:.4f} Myr'.format(t_dyn, t_cgs))
print(' soft = {:.4f} r_IU = {:.4f} pc'.format(soft, soft*R0))
print()

############################################################################################################################

# FUNCTIONS

# The tree inverse CDFs for the homogeneous sphere:

def R_P(P,a):
	return a * np.cbrt(P)

def Ph_P(P):
	return 2 * np.pi * P

def Th_P(P):
	return np.arccos(1. - 2. * P)

# The tree PDFs for the homogeneous sphere:

def pdf_r(r,a):
	return 3. * r**2 / (a**3)

def pdf_ph(ph):
	return 0.5 / np.pi * (1. + 0. * ph)

def pdf_th(th):
	return 0.5 * np.sin(th)

# The circular velocity of the Plummer sphere, and derivative formulae

def v_circ(r):
	return r * np.sqrt( rho * 4. * np.pi / 3. )

def v_case_1(r):
	return 0.05 * v_circ(r)

def v_case_2(r):
	return 0.05 * a * v_circ(a) / r

def pdf_v_1(v):
	k = 0.05 * np.sqrt( rho * 4. * np.pi / 3. )
	return 3. / ( (k * a * V0)**3 ) * v**2 

def pdf_v_2(v):
	k = 0.05 * np.sqrt( rho * 4. * np.pi / 3. ) * (a**2) * V0 * R0
	return 3. * (k**3) / (a**3 * R0**3) / (v**4)

def pdf_l_1(l):
	k = 0.05 * np.sqrt( rho * 4. * np.pi / 3. ) * V0 / R0
	return 1.5 / (a**3 * R0**3) / (k**(3./2.)) * l**(1./2.) 

# A plotting primer for PDFs over histograms

def histo_pdf_plotter(ax, x_lim, x_step, x_bins, func, npar, x_min=0.):
	x = np.linspace(x_min, x_lim, 10000)
	if npar == 1:
		f_x = func(x, a * R0)
		ax.plot(x, f_x, color='black', lw=0.9)
	elif npar == 0:
		f_x = func(x)
		ax.plot(x, f_x, color='black', lw=0.9)
	for j in range(len(x_bins)):
		x = []
		f_x = []
		e_x = []
		x_mid = x_bins[j] + x_step / 2
		for i in range(9):
			x_temp = x_bins[j] + x_step / 2 + x_step / 15 * (i-4)
			if npar == 1:
				f_temp = func(x_mid, a * R0)
			elif npar == 0:
				f_temp = func(x_mid)
			e_temp = np.sqrt(f_temp * N) / N
			ax.vlines(x_mid, ymin=f_temp-e_temp, ymax=f_temp+e_temp, color='black', lw=0.9)
			x.append(x_temp)
			f_x.append(f_temp)
			e_x.append(e_temp)
		x = np.array(x)
		f_x = np.array(f_x)
		e_x = np.array(e_x)
		ax.plot(x, f_x + e_x, color='black', lw=0.9)
		ax.plot(x, f_x - e_x, color='black', lw=0.9)

def histo_pdf_plotter_log(ax, x_min, x_max, x_step, x_bins, func, npar):
	log_min = np.log10(x_min)
	log_max = np.log10(x_max)
	x = np.logspace(log_min, log_max, 1000)
	if npar == 1:
		f_x = func(x, a * R0) # * N
		ax.plot(x, f_x, color='black', lw=0.9)
	elif npar == 0:
		f_x = func(x) # * N
		ax.plot(x, f_x, color='black', lw=0.9)
	for j in range(len(x_bins)-1):
		x = []
		f_x = []
		e_x = []
		x_mid = (x_bins[j+1] + x_bins[j]) / 2.
		x_step = x_bins[j+1] - x_bins[j]
		for i in range(9):
			x_temp = x_mid + x_step / 15 * (i-4)
			if npar == 1:
				f_temp = func(x_mid, a * R0) #* N
			elif npar == 0:
				f_temp = func(x_mid) # * N
			e_temp = np.sqrt(f_temp * N) / N # np.sqrt(f_temp) # 
			ax.vlines(x_mid, ymin=f_temp-e_temp, ymax=f_temp+e_temp, color='black', lw=0.9)
			x.append(x_temp)
			f_x.append(f_temp)
			e_x.append(e_temp)
		x = np.array(x)
		f_x = np.array(f_x)
		e_x = np.array(e_x)
		ax.plot(x, f_x + e_x, color='black', lw=0.9)
		ax.plot(x, f_x - e_x, color='black', lw=0.9)

############################################################################################################################

M = np.full(shape=N, fill_value=M_tot/N)

P_R  = np.random.uniform(size=N)
P_Ph = np.random.uniform(size=N)
P_Th = np.random.uniform(size=N)

R = np.zeros(shape=(4,N))

R[0,:] = R_P(P_R, a)
Ph = Ph_P(P_Ph)
Th = Th_P(P_Th)

R[1,:] = R[0] * np.sin(Th) * np.cos(Ph)
R[2,:] = R[0] * np.sin(Th) * np.sin(Ph)
R[3,:] = R[0] * np.cos(Th)

V_1    = np.zeros(shape=(4,N))
V_Ph_1 = np.zeros(shape=N)
V_Th_1 = np.zeros(shape=N)
V_2    = np.zeros(shape=(4,N))
V_Ph_2 = np.zeros(shape=N)
V_Th_2 = np.zeros(shape=N)

for i in range(N):

	r  = R[0,i]
	ph = Ph[i]
	th = Th[i]

	V_1[0,i] = 0.05 * v_circ(r)
	V_2[0,i] = 0.05 * v_circ(a) * a / r

	P_b = np.random.uniform()
	b = Th_P(P_b)

	V_1[1,i] = V_1[0,i] * ( np.cos(b) * np.cos(ph) * np.cos(th) - np.sin(b) * np.sin(ph) )
	V_1[2,i] = V_1[0,i] * ( np.cos(b) * np.sin(ph) * np.cos(th) - np.sin(b) * np.cos(ph) )
	V_1[3,i] = V_1[0,i] * ( - np.cos(b) * np.sin(th) )
	
	V_2[1,i] = V_2[0,i] * ( np.cos(b) * np.cos(ph) * np.cos(th) - np.sin(b) * np.sin(ph) )
	V_2[2,i] = V_2[0,i] * ( np.cos(b) * np.sin(ph) * np.cos(th) - np.sin(b) * np.cos(ph) )
	V_2[3,i] = V_2[0,i] * ( - np.cos(b) * np.sin(th) )

	ph_1 = np.arctan2( V_1[2,i] , V_1[1,i])
	ph_2 = np.arctan2( V_2[2,i] , V_2[1,i])
	if ph_1 < 0.:
		ph_1 += 2. * np.pi
	if ph_2 < 0.:
		ph_2 += 2. * np.pi
	V_Ph_1[i] = ph_1
	V_Ph_2[i] = ph_2
	V_Th_1[i] = np.arccos( V_1[3,i] / V_1[0,i])
	V_Th_2[i] = np.arccos( V_2[3,i] / V_2[0,i])

L_1 = np.zeros(shape=(4,N))
L_2 = np.zeros(shape=(4,N))

C_1 = np.zeros(shape=N)
C_2 = np.zeros(shape=N)

for i in range(N):

	L_1[1,i] = R[2,i] * V_1[3,i] - R[3,i] * V_1[2,i]
	L_1[2,i] = R[3,i] * V_1[1,i] - R[1,i] * V_1[3,i]
	L_1[3,i] = R[1,i] * V_1[2,i] - R[2,i] * V_1[1,i]
	L_1[0,i] = np.sqrt( L_1[1,i]**2 + L_1[2,i]**2 + L_1[3,i]**2 )

	L_2[1,i] = R[2,i] * V_2[3,i] - R[3,i] * V_2[2,i]
	L_2[2,i] = R[3,i] * V_2[1,i] - R[1,i] * V_2[3,i]
	L_2[3,i] = R[1,i] * V_2[2,i] - R[2,i] * V_2[1,i]
	L_2[0,i] = np.sqrt( L_2[1,i]**2 + L_2[2,i]**2 + L_2[3,i]**2 )

	for j in range(3):
		C_1[i] += R[j+1,i] * V_1[j+1,i] 
		C_2[i] += R[j+1,i] * V_2[j+1,i] 
	C_1[i] = C_1[i] / (R[0,i] * V_1[0,i] )
	C_2[i] = C_2[i] / (R[0,i] * V_2[0,i] )

C_1 = np.arccos( C_1 )
C_2 = np.arccos( C_2 )

# print(len(C_1))
# print(len(C_1[C_1 != 0.]))
# print(len(C_1[C_1 < - np.pi]))
# print(len(C_1[C_1 > + np.pi]))

print('Inertial RF:')
print('max C_1 = {:}'.format(np.amax(C_1)))
print('min C_1 = {:}'.format(np.amin(C_1)))
print('max C_2 = {:}'.format(np.amax(C_2)))
print('min C_2 = {:}'.format(np.amin(C_2)))

############################################################################################################################

# CREATE FILES WITH INITIAL CONDITIONS

if save == 'save=Y':
	file = open("IC_OCT_Exam_C1_{:d}.txt".format(N), "w")
	file.write("{:d}\n".format(N))
	file.write("{:d}\n".format(3))
	file.write("{:3f}\n".format(0.))
	for i in range(N):
		file.write("{:f}\n".format(M[i]))
	for i in range(N):
		file.write("{:f} {:f} {:f}\n".format(R[1,i], R[2,i], R[3,i]))
	for i in range(N):
		file.write("{:f} {:f} {:f}\n".format(V_1[1,i], V_1[2,i], V_1[3,i]))
	file.close()

	file = open("IC_OCT_Exam_C2_{:d}.txt".format(N), "w")
	file.write("{:d}\n".format(N))
	file.write("{:d}\n".format(3))
	file.write("{:3f}\n".format(0.))
	for i in range(N):
		file.write("{:f}\n".format(M[i]))
	for i in range(N):
		file.write("{:f} {:f} {:f}\n".format(R[1,i], R[2,i], R[3,i]))
	for i in range(N):
		file.write("{:f} {:f} {:f}\n".format(V_2[1,i], V_2[2,i], V_2[3,i]))
	file.close()

	file = open("IC_OCT_Exam_C3_{:d}.txt".format(N), "w")
	file.write("{:d}\n".format(N))
	file.write("{:d}\n".format(3))
	file.write("{:3f}\n".format(0.))
	for i in range(N):
		file.write("{:f}\n".format(M[i]))
	for i in range(N):
		file.write("{:f} {:f} {:f}\n".format(R[1,i], R[2,i], R[3,i]))
	for i in range(N):
		file.write("{:f} {:f} {:f}\n".format(0.,0.,0.))
	file.close()

############################################################################################################################

# HISTOGRAMS WITH POSITION-SPACE SAMPLING

r_lim   = a * R0 
r_step  = r_lim / 40.
ph_lim  = 2. * np.pi
ph_step = np.pi / 12.
th_lim  = np.pi
th_step = np.pi / 12.

fig_h = plt.figure(figsize=(7,6), constrained_layout=True)
gs = GridSpec(2, 3, figure=fig_h)
ax_h_R  = fig_h.add_subplot(gs[0,0:3])
ax_h_Ph = fig_h.add_subplot(gs[1,0:2])
ax_h_Th = fig_h.add_subplot(gs[1,2:3])

R_bins = np.linspace(start=0, stop=r_lim+0.1*r_step, num=40) #, step=r_step)
ax_h_R.hist(R[0] * R0, bins=R_bins, color='lightgrey', alpha=1, edgecolor='black', density=True)
histo_pdf_plotter(ax=ax_h_R, x_lim=r_lim, x_step=r_step, x_bins=R_bins, func=pdf_r, npar=1)
ax_h_R.set_title('Position-space'+'\n'+'Radius pdf')
ax_h_R.set_xlabel(r'$r\;$[pc]')
ax_h_R.set_xlim(0,r_lim)
ax_h_R.set_ylim(0,None)
ax_h_R.grid(ls=':',which='both')

Th_bins = np.linspace(start=0, stop=th_lim+0.1*th_step, num=12) # , step=th_step)
ax_h_Th.hist(Th, bins=Th_bins, color='lightgrey', alpha=1, edgecolor='black', density=True)
histo_pdf_plotter(ax=ax_h_Th, x_lim=th_lim, x_step=th_step, x_bins=Th_bins, func=pdf_th, npar=0)
ax_h_Th.set_title('Position-space'+'\n'+'Polar angle pdf')
ax_h_Th.set_xlabel(r'$\vartheta \;$[rad]')
ax_h_Th.set_xlim(0,th_lim)
ax_h_Th.xaxis.set_major_locator(tck.MultipleLocator(np.pi / 4))
ax_h_Th.xaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.2f}$\,\pi$'.format(val/np.pi) if val !=0 else '0'))
ax_h_Th.grid(ls=':',which='both')

Ph_bins = np.linspace(start=0,stop=ph_lim+0.1*ph_step, num=24) #,step=ph_step)
ax_h_Ph.hist(Ph, bins=Ph_bins, color='lightgrey', alpha=1, edgecolor='black', density=True)
histo_pdf_plotter(ax=ax_h_Ph, x_lim=ph_lim, x_step=ph_step, x_bins=Ph_bins, func=pdf_ph, npar=0)
ax_h_Ph.set_title('Position-space'+'\n'+'Azimuthal angle pdf')
ax_h_Ph.set_xlabel(r'$\varphi \;$[rad]')
ax_h_Ph.set_xlim(0,ph_lim)
ax_h_Ph.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
ax_h_Ph.xaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.2f}$\,\pi$'.format(val/np.pi) if val !=0 else '0'))
ax_h_Ph.grid(ls=':',which='both')

############################################################################################################################

# HISTOGRAMS WITH VELOCITY-SPACE SAMPLING

fig_VV , ax_VV = plt.subplots(2,1, figsize=(7,6))

v_lim   = 0.05 * v_circ(a) * V0
v_step = v_lim / 40.
v_2_max = np.amax(V_2[0]) * V0		# v_2_max = 1. *
v_2_min = np.amin(V_2[0]) * V0
v_step_2 = (v_2_max - v_2_min) / 40. 

fig_VV.suptitle('\n'+'Initial velocity module - pdf')
V_1_bins = np.linspace(start=0.,stop=v_lim, num=41) #,step=v_step)
ax_VV[0].hist(V_1[0] * V0, bins=V_1_bins, color='lightgrey', alpha=1, edgecolor='black', density=True)
histo_pdf_plotter(ax=ax_VV[0], x_lim=v_lim, x_step=v_step, x_bins=V_1_bins, func=pdf_v_1, npar=0)
ax_VV[0].set_title('Case 1')
ax_VV[0].set_xlabel(r'$v\;$[km/s]',)
ax_VV[0].set_xlim(0,v_lim)
ax_VV[0].set_ylim(0,None)
ax_VV[0].grid(ls=':',which='both')

V_2_bins = np.logspace(start=np.log10(v_2_min), stop=np.log10(v_2_max), num=41)
# V_2_bins = np.linspace(start=v_2_min, stop=v_2_max, num=41)
ax_VV[1].hist(V_2[0] * V0, bins=V_2_bins, color='lightgrey', alpha=1, edgecolor='black', density=True)
histo_pdf_plotter_log(ax=ax_VV[1], x_min=v_2_min, x_max=v_2_max, x_step=v_step_2, x_bins=V_2_bins, func=pdf_v_2, npar=0)
# histo_pdf_plotter(ax=ax_VV[1], x_min=v_2_min, x_lim=v_2_max, x_step=v_step_2, x_bins=V_2_bins, func=pdf_v_2, npar=0)
ax_VV[1].set_title('Case 2')
ax_VV[1].set_xlabel(r'$v\;$[km/s]',)
ax_VV[1].set_xlim(v_2_min,v_2_max)
ax_VV[1].set_xscale('log')
ax_VV[1].set_yscale('log')
ax_VV[1].set_ylim(None,None)
ax_VV[1].grid(ls=':',which='both')

fig_VV.tight_layout()

############################################################################################################################

# HISTOGRAMS WITH ANGULAR MOMENTA

fig_LL , ax_LL = plt.subplots(2,1, figsize=(7,6))

fig_LL.suptitle('\n'+'Initial angular momentum module - pdf')

l_max  = np.amax(L_1) * R0 * V0
l_step = l_max / 40
L_1_bins = np.linspace(start=0, stop=l_max, num=41)
ax_LL[0].hist(L_1[0] * V0 * R0, bins=L_1_bins, color='lightgrey', alpha=1, edgecolor='black', density=True)
# histo_pdf_plotter(ax=ax_LL[0], x_lim=l_max, x_step=l_step, x_bins=L_1_bins, func=pdf_l_1, npar=0)
ax_LL[0].set_title('Case 1')

l_max  = np.amax(L_2) * R0 * V0
l_step = l_max / 40
l_exp  = 0.05 * (a * R0) * (v_circ(a) * V0)
L_2_bins = np.linspace(start=0, stop=l_max, num=41)
ax_LL[1].hist(L_2[0] * V0 * R0, bins=L_2_bins, color='lightgrey', alpha=1, edgecolor='black', density=True)
# histo_pdf_plotter(ax=ax_v_1_V, x_lim=l_max, x_step=l_step, x_bins=L_1_bins, func=pdf_l_2, npar=0)
ax_LL[1].set_title('Case 2')

for i in range(2):
	ax_LL[i].set_xlabel(r'$l\;$[pc km s$^{-1}$]',)
	ax_LL[i].set_xlim(0,l_max)
	ax_LL[i].set_ylim(0,None)
	ax_LL[i].grid(ls=':',which='both')

y_m , y_M = ax_LL[1].get_ylim()
ax_LL[1].plot([l_exp, l_exp],[y_m,y_M], color='black', ls='--', lw=0.9)

fig_LL.tight_layout()

############################################################################################################################

# SCATTER PLOTS WITH ANGULAR MOMENTA, RV ANGLES

# fig_L , ax_L = plt.subplots(2,1,figsize=(7,7))
fig_L = plt.figure(figsize=(7,6),constrained_layout=True)
gs = GridSpec(2, 2, figure=fig_L)
ax_L = []
ax_A = []
ax_L.append( fig_L.add_subplot(gs[0,0:1]) )
ax_L.append( fig_L.add_subplot(gs[1,0:1]) )
ax_A.append( fig_L.add_subplot(gs[0,1:2]) )
ax_A.append( fig_L.add_subplot(gs[1,1:2]) )

sort_index = np.argsort(R[0])
ax_L[0].plot(R[0,sort_index] * R0, R0 * V0 * L_1[0,sort_index], color='black', ls='', marker='o', ms=0.1, label='Case 1:   ' + r'$ |\vec{l}| = |\vec{r} \times \vec{v}| $')	# , ls=':',  				   
ax_L[1].plot(R[0,sort_index] * R0, R0 * V0 * L_2[0,sort_index], color='black', ls='', marker='o', ms=0.1, label='Case 2:   ' + r'$ |\vec{l}| = |\vec{r} \times \vec{v}| $')	# , ls='--', 				   
ax_A[0].plot(R[0,sort_index] * R0, C_1[sort_index], color='black', ls='', marker='o', ms=0.1, label='Case 1:   ' + r'$ \vartheta_{\hat{rv}} = \cos^{-1} \frac{\vec{r} \cdot \vec{v}}{r \, v} $')	# , ls='-.', 				   
ax_A[1].plot(R[0,sort_index] * R0, C_2[sort_index], color='black', ls='', marker='o', ms=0.1, label='Case 2:   ' + r'$ \vartheta_{\hat{rv}} = \cos^{-1} \frac{\vec{r} \cdot \vec{v}}{r \, v} $')	# , ls=(0, (3, 5, 1, 5, 1, 5)), 

for i in range(2):

	ax_L[i].grid(ls=':', which='both')
	ax_L[i].legend(frameon=True)
	ax_L[i].set_xlim(0,r_lim)
	ax_L[i].set_xlabel(r'$r\;$[pc]')
	ax_L[i].set_ylabel(r'$l\;$[pc km s$^{-1}$]')

	ax_A[i].grid(ls=':',which='both')
	ax_A[i].legend(frameon=True)
	ax_A[i].set_ylim(0,+th_lim)
	ax_A[i].set_xlim(0,r_lim)
	ax_A[i].xaxis.set_major_locator(tck.MultipleLocator(1.))
	ax_A[i].yaxis.set_major_locator(tck.MultipleLocator(np.pi/4))
	ax_A[i].yaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.2f}$\,\pi$'.format(val/np.pi) if val !=0 else '0'))
	ax_A[i].set_xlabel(r'$r\;$[pc]')
	ax_A[i].set_ylabel('\n'+r'$\vartheta_{\hat{rv}}\;$[rad]')

############################################################################################################################

# EVALUATE CM OF THE SYSTEM, REDO SCATTER PLOTS

R_cm   = np.zeros(shape=4)
V_1_cm = np.zeros(shape=4)
V_2_cm = np.zeros(shape=4)

for i in range(N):
	for j in range(3):
		R_cm[j+1]   += R[j+1,i]   / N
		V_1_cm[j+1] += V_1[j+1,i] / N
		V_2_cm[j+1] += V_2[j+1,i] / N
	R_cm[0]   = np.sqrt( R_cm[1]**2   + R_cm[2]**2   + R_cm[3]**2 )
	V_1_cm[0] = np.sqrt( V_1_cm[1]**2 + V_1_cm[2]**2 + V_1_cm[3]**2 )
	V_2_cm[0] = np.sqrt( V_2_cm[1]**2 + V_2_cm[2]**2 + V_2_cm[3]**2 )

for i in range(N):

	for j in range(3):
		R[j+1,i]   = R[j+1,i]   - R_cm[j+1]
		V_1[j+1,i] = V_1[j+1,i] - V_1_cm[j+1]
		V_2[j+1,i] = V_2[j+1,i] - V_2_cm[j+1]

	R[0,i] = np.sqrt( R[1,i]**2 + R[2,i]**2 + R[3,i]**2 )
	V_1[0,i] = np.sqrt( V_1[1,i]**2 + V_1[2,i]**2 + V_1[3,i]**2 )
	V_2[0,i] = np.sqrt( V_2[1,i]**2 + V_2[2,i]**2 + V_2[3,i]**2 )

	L_1[1,i] = R[2,i] * V_1[3,i] - R[3,i] * V_1[2,i]
	L_1[2,i] = R[3,i] * V_1[1,i] - R[1,i] * V_1[3,i]
	L_1[3,i] = R[1,i] * V_1[2,i] - R[2,i] * V_1[1,i]
	L_1[0,i] = np.sqrt( L_1[1,i]**2 + L_1[2,i]**2 + L_1[3,i]**2 )

	L_2[1,i] = R[2,i] * V_2[3,i] - R[3,i] * V_2[2,i]
	L_2[2,i] = R[3,i] * V_2[1,i] - R[1,i] * V_2[3,i]
	L_2[3,i] = R[1,i] * V_2[2,i] - R[2,i] * V_2[1,i]
	L_2[0,i] = np.sqrt( L_2[1,i]**2 + L_2[2,i]**2 + L_2[3,i]**2 )

	for j in range(3):
		C_1[i] += R[j+1,i] * V_1[j+1,i] / (R[0,i] * V_1[0,i])
		C_2[i] += R[j+1,i] * V_2[j+1,i] / (R[0,i] * V_2[0,i])

print('max C_1 = {:}   --->   {:}'.format(np.amax(C_1), np.arccos(np.amax(C_1))))
print('min C_1 = {:}   --->   {:}'.format(np.amin(C_1), np.arccos(np.amin(C_1))))
print('max C_2 = {:}   --->   {:}'.format(np.amax(C_2), np.arccos(np.amax(C_2))))
print('min C_2 = {:}   --->   {:}'.format(np.amin(C_2), np.arccos(np.amin(C_2))))

C_1[i] = np.arccos( C_1[i] )
C_2[i] = np.arccos( C_2[i] )

print('CM RF:')
print('max C_1 = {:}'.format(np.amax(C_1)))
print('min C_1 = {:}'.format(np.amin(C_1)))
print('max C_2 = {:}'.format(np.amax(C_2)))
print('min C_2 = {:}'.format(np.amin(C_2)))

# fig_LCM , ax_LCM = plt.subplots(2,1,figsize=(7,7))
fig_LCM = plt.figure(figsize=(7,6),constrained_layout=True)
gs = GridSpec(2, 2, figure=fig_LCM)
ax_LCM = []
ax_ACM = []
ax_LCM.append( fig_LCM.add_subplot(gs[0,0:1]) )
ax_LCM.append( fig_LCM.add_subplot(gs[1,0:1]) )
ax_ACM.append( fig_LCM.add_subplot(gs[0,1:2]) )
ax_ACM.append( fig_LCM.add_subplot(gs[1,1:2]) )

sort_index = np.argsort(R[0])
ax_LCM[0].plot(R[0,sort_index] * R0, R0 * V0 * L_1[0,sort_index], color='black' , ls='', marker='o', ms=0.1, label='Case 1:   ' + r'$ |\vec{l}| = |\vec{r} \times \vec{v}| $')	# , ls=':',  				   
ax_LCM[1].plot(R[0,sort_index] * R0, R0 * V0 * L_2[0,sort_index], color='black' , ls='', marker='o', ms=0.1, label='Case 2:   ' + r'$ |\vec{l}| = |\vec{r} \times \vec{v}| $')	# , ls='--', 				   
ax_ACM[0].plot(R[0,sort_index] * R0, C_1[sort_index], color='black', ls='', marker='o', ms=0.1, label='Case 1:   ' + r'$ \vartheta_{\hat{rv}} = \cos^{-1} \frac{\vec{r} \cdot \vec{v}}{r \, v} $')					# , ls='-.', 				   
ax_ACM[1].plot(R[0,sort_index] * R0, C_2[sort_index], color='black', ls='', marker='o', ms=0.1, label='Case 2:   ' + r'$ \vartheta_{\hat{rv}} = \cos^{-1} \frac{\vec{r} \cdot \vec{v}}{r \, v} $')					# , ls=(0, (3, 5, 1, 5, 1, 5)), 

for i in range(2):
	ax_LCM[i].grid(ls=':', which='both')
	ax_LCM[i].legend(frameon=True)
	ax_LCM[i].set_xlim(0,r_lim)
	ax_LCM[i].set_xlabel(r'$r\;$[pc]')
	ax_LCM[i].set_ylabel(r'$l\;$[pc km s$^{-1}$]')

	ax_ACM[i].grid(ls=':',which='both')
	ax_ACM[i].legend(frameon=True)
	ax_ACM[i].set_xlim(0,r_lim)
	ax_ACM[i].set_ylim(0,+th_lim)
	ax_ACM[i].xaxis.set_major_locator(tck.MultipleLocator(1.))
	ax_ACM[i].yaxis.set_major_locator(tck.MultipleLocator(np.pi/4))
	ax_ACM[i].yaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.2f}$\,\pi$'.format(val/np.pi) if val !=0 else '0'))
	ax_ACM[i].set_xlabel(r'$r\;$[pc]')
	ax_ACM[i].set_ylabel('\n'+r'$\vartheta_{\hat{rv}}\;$[rad]')

############################################################################################################################

# HISTORGAMS WITH RV ANGLES

fig_CC , ax_CC = plt.subplots(1,2, figsize=(7,4), tight_layout=True)

c_max = 0.65 * np.pi 
c_min = 0.35 * np.pi
c_step = np.pi / 8.
C_bins = np.linspace(start=c_min, stop=c_max, num=24)

fig_CC.suptitle('\n'+'Angle between initial position and velocity - pdf')

c1_height , c1_th , c1_patches = ax_CC[0].hist(C_1, bins=C_bins, color='lightgrey', alpha=1, edgecolor='black', density=True)
ax_CC[0].set_title('Case 1')

c2_height , c2_th , c2_patches = ax_CC[1].hist(C_2, bins=C_bins, color='lightgrey', alpha=1, edgecolor='black', density=True)
ax_CC[1].set_title('Case 2')

fit = fitting.LevMarLSQFitter()
gaus_1 = models.Gaussian1D(amplitude=np.amax(c1_height), mean=0.5*np.pi, stddev=0.05*np.pi)
gaus_2 = models.Gaussian1D(amplitude=np.amax(c2_height), mean=0.5*np.pi, stddev=0.05*np.pi)
th_fit = 0.5 * (c1_th[1:] + c1_th[:-1])  # np.linspace(c_min,c_max,len(c1_height))
th_plot = np.linspace(c_min,c_max,10000)
gausfit_1 = fit(gaus_1, th_fit, c1_height)
# ax_CC[0].plot(th_plot, gausfit_1(th_plot), ls=':', lw=0.9, color='black')
gausfit_2 = fit(gaus_2, th_fit, c2_height)
# ax_CC[1].plot(th_plot, gausfit_2(th_plot), ls=':', lw=0.9, color='black')

c1_stddev = 1. * gausfit_1.stddev # * 2.355 se fwhm 
c2_stddev = 1. * gausfit_2.stddev
print('c1_stddev = {:.3f} pi'.format(c1_stddev / np.pi))
print('c2_stddev = {:.3f} pi'.format(c2_stddev / np.pi))

moff_1 = models.Moffat1D(amplitude=np.amax(c1_height), x_0=0.5*np.pi, gamma=1, alpha=1)
moff_2 = models.Moffat1D(amplitude=np.amax(c2_height), x_0=0.5*np.pi, gamma=1, alpha=1)
th_fit = 0.5 * (c1_th[1:] + c1_th[:-1])  # np.linspace(c_min,c_max,len(c1_height))
th_plot = np.linspace(c_min,c_max,10000)
moffit_1 = fit(moff_1, th_fit, c1_height)
ax_CC[0].plot(th_plot, moffit_1(th_plot), ls='--', lw=0.9, color='black')
moffit_2 = fit(moff_2, th_fit, c2_height)
ax_CC[1].plot(th_plot, moffit_2(th_plot), ls='--', lw=0.9, color='black')

# fig_CC.tight_layout()

for i in range(2):
	ax_CC[i].grid(ls=':',which='both')
	ax_CC[i].set_xlim(c_min,c_max)
	# ax_CC[i].xaxis.set_major_locator(tck.MultipleLocator(0.15 / 2 * np.pi))
	ax_CC[i].set_xticks(np.array([0.350, 0.425, 0.500, 0.575, 0.650]) * np.pi)
	ax_CC[i].xaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.3f}$\,\pi$'.format(val/np.pi) if val !=0 else '0'))
	ax_CC[i].set_xlabel(r'$\vartheta_{\hat{rv}}\;$[rad]'+'\n')
	ax_CC[i].set_ylabel('\n\n ')

############################################################################################################################

# SAVE ALL FIGURES

if save == 'save=Y':
	fig_CC.savefig("Results_PNG/IC_Exam_Angle-rv.png".format(N), bbox_inches='tight', dpi=400)
	fig_CC.savefig("Results_PNG/IC_Exam_Angle-rv.eps".format(N), bbox_inches='tight')
	fig_LL.savefig("Results_PNG/IC_Exam_L-space.png".format(N), bbox_inches='tight', dpi=400)
	fig_LL.savefig("Results_PNG/IC_Exam_L-space.eps".format(N), bbox_inches='tight')
	fig_L.savefig("Results_PNG/IC_Exam_L-scatter.png".format(N), bbox_inches='tight', dpi=400)
	fig_L.savefig("Results_PNG/IC_Exam_L-scatter.eps".format(N), bbox_inches='tight')
	fig_LCM.savefig("Results_PNG/IC_Exam_L-scatter-CM.png".format(N), bbox_inches='tight', dpi=400)
	fig_LCM.savefig("Results_PNG/IC_Exam_L-scatter-CM.eps".format(N), bbox_inches='tight')
	fig_h.savefig("Results_PNG/IC_Exam_R-space.png".format(N), bbox_inches='tight', dpi=400)
	fig_h.savefig("Results_PNG/IC_Exam_R-space.eps".format(N), bbox_inches='tight')
	fig_VV.savefig("Results_PNG/IC_Exam_V-space.png".format(N), bbox_inches='tight', dpi=400)
	fig_VV.savefig("Results_PNG/IC_Exam_V-space.eps".format(N), bbox_inches='tight')


############################################################################################################################

print()
plt.show()


