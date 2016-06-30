from matplotlib import pyplot as plt
import numpy as np

def data_reader(campaigns):
    '''
    reads the TCXX_campaign_XX.txt files and returns numpy arrays

    iput:   string - campaign combination file name  

    output: (2D array, 1D array) - (test parameters and number of cracks,
                                    chamber parameters)
    '''
    campaign_data_lst = []
    chamber_data_lst = []
    for campaign in campaigns:
        # loading the output file as a numpy array
        # 'data_pack/'
        data_pack_dir = '/Users/rostislavrypl/git/rostar/survival_probability/data_pack/'
        campaign_data = np.loadtxt(
            data_pack_dir + 'TC_campaign_combinations/' + campaign + '.txt', skiprows=8)
        campaign_data_lst.append(campaign_data)
        chamber_data = np.loadtxt(
            data_pack_dir + 'TCs/' + campaign[0:4] + '.txt', skiprows=13)
        chamber_data_lst.append(chamber_data)
    return campaign_data_lst

data = data_reader(['TC30_campaign_01', 'TC30_campaign_02', 'TC30_campaign_03', 'TC30_campaign_04', 'TC30_campaign_05',
                   'TC30_campaign_06', 'TC30_campaign_07', 'TC30_campaign_08', 'TC30_campaign_09','TC30_campaign_10'])

camps = np.arange(9) + 1

m = np.array([4.8,5.26,9.82,9.85,9.89,5.64,5.8,5.14,5.09])
s = np.array([4500, 6825, 8350, 9267,8032,7816,8433,9034,9487])
mr= np.array([-0.023,0.041,-0.031,-0.136,0.35,-0.45,-0.43,-0.43,-0.486])
p = np.array([0.002, 0.0025,-0.057, 0.0160, 0.025,-0.027,-0.074,-0.028,-0.029])
ft = np.array([-0.085,0.0082,-0.024, -0.019,-0.028,-0.026,-0.029,-0.026,-0.008])
cyc =np.array([3.1,4.07,3.55,2.88,4.12,0.334,0.36,0.19,0.227])
p2 = np.array([-0.0001, -0.00036, 0.00041, -0.00041, -0.00044, 0.00046,-0.00017, -3.66e-5,5.88e-5])
mr2 = np.array([0.001,-0.0051,0.012,0.0295,0.0193,-0.0156,0.0066,0.0445,0.021])
ft2 = np.array([-0.00035, -4.37e-5, 0.00018, 0.00019, 0.000265, -1.83e-5, -1.94e-5, -1.9e-5, -2.13e-5])
pmr = np.array([-0.0069, 0.002, 0.0060, 0.0091, 0.00142, 0.0008, 0.0095, 0.0094, 0.0095])
pft = np.array([0.00073, 0.00026, -0.00045, 0.00041, 0.00022, 0.000028, 0.00039, 0.00028, 0.00015])
mrft = np.array([-0.00076, -0.0054, 0.001, -0.017, -0.015, 0.0076, -0.00082, 0.00079, 0.0003])

camp_sets = []
ft_effect = []
minft = []
maxft = []
yft = []
p_effect = []
minp = []
maxp = []
yp = []
mr_effect = []
minmr = []
maxmr = []
ymr = []
p2_effect = []
minp2 = []
maxp2 = []
yp2 = []
ft2_effect = []
minft2 = []
maxft2 = []
yft2 = []
mr2_effect = []
minmr2 = []
maxmr2 = []
ymr2 = []

pft_effect = []
minpft = []
maxpft = []
ypft = []
ftmr_effect = []
minftmr = []
maxftmr = []
yftmr = []
mrp_effect = []
minmrp = []
maxmrp = []
ymrp = []


for i, camp in enumerate(data[:-1]):
    ft_data = np.array([])
    p_data = np.array([])
    mr_data = np.array([])
    for j in range(i+1):
        ft_data = np.hstack((ft_data,data[j][:,1]))
        p_data = np.hstack((p_data,data[j][:,2]))
        mr_data = np.hstack((mr_data,data[j][:,3]))
    ft_effect.append(np.exp(ft_data*ft[i]))
    p_effect.append(np.exp(p_data*p[i]))
    mr_effect.append(np.exp(mr_data*mr[i]))
    p2_effect.append(np.exp(p_data**2*p2[i]))
    ft2_effect.append(np.exp(ft_data**2*ft2[i]))
    mr2_effect.append(np.exp(mr_data**2*mr2[i]))
    pft_effect.append(np.exp(p_data*ft_data*pft[i]))
    ftmr_effect.append(np.exp(ft_data*mr_data*mrft[i]))
    mrp_effect.append(np.exp(mr_data*p_data*pmr[i]))
 
full_ft = []
full_p = []
full_mr = []
full_ft2 = []
full_p2 = []
full_mr2 = []
full_mrp = []
full_pft = []
full_ftmr = []
for i, camp in enumerate(data[:-1]):
    yft.append(np.mean(ft_effect[i]))
    minft.append(np.min(ft_effect[i]))
    maxft.append(np.max(ft_effect[i]))

    yp.append(np.mean(p_effect[i]))
    minp.append(np.min(p_effect[i]))
    maxp.append(np.max(p_effect[i]))
    
    ymr.append(np.mean(mr_effect[i]))
    minmr.append(np.min(mr_effect[i]))
    maxmr.append(np.max(mr_effect[i]))

    yft2.append(np.mean(ft2_effect[i]))
    minft2.append(np.min(ft2_effect[i]))
    maxft2.append(np.max(ft2_effect[i]))

    yp2.append(np.mean(p2_effect[i]))
    minp2.append(np.min(p2_effect[i]))
    maxp2.append(np.max(p2_effect[i]))
    
    ymr2.append(np.mean(mr2_effect[i]))
    minmr2.append(np.min(mr2_effect[i]))
    maxmr2.append(np.max(mr2_effect[i]))

    ypft.append(np.mean(pft_effect[i]))
    minpft.append(np.min(pft_effect[i]))
    maxpft.append(np.max(pft_effect[i]))

    ymrp.append(np.mean(mrp_effect[i]))
    minmrp.append(np.min(mrp_effect[i]))
    maxmrp.append(np.max(mrp_effect[i]))
    
    yftmr.append(np.mean(ftmr_effect[i]))
    minftmr.append(np.min(ftmr_effect[i]))
    maxftmr.append(np.max(ftmr_effect[i]))

    full_ft.append(ft_effect[i])
    full_p.append(p_effect[i])
    full_mr.append(mr_effect[i])
    full_ft2.append(ft2_effect[i])
    full_p2.append(p2_effect[i])
    full_mr2.append(mr2_effect[i])
    full_mrp.append(mrp_effect[i])
    full_pft.append(pft_effect[i])
    full_ftmr.append(ftmr_effect[i])

x = np.linspace(1,9,9)
# if True:
#     plt.plot(x,yp, lw=2, label='pressure', color='green')
#     #plt.fill_between(x, minp, maxp, alpha=0.3, color='green')
#     plt.plot(x,,yft, lw=2, label='firing times', color='blue')
#     #plt.fill_between(x, minft, maxft, alpha=0.3, color='blue')
#     plt.plot(x,ymr, lw=2, label='mixture ratio', color='red')
#     #plt.fill_between(x, minmr, maxmr, alpha=0.3, color='red')
# 
# if True:
#     plt.plot(x,yp2, lw=2, label='pressure2', color='yellow')
#     #plt.fill_between(x, minp2, maxp2, alpha=0.3, color='yellow')
#     plt.plot(x,yft2, lw=2, label='firing times2', color='magenta')
#     #plt.fill_between(x, minft2, maxft2, alpha=0.3, color='magenta')
#     plt.plot(x,ymr2, lw=2, label='mixture ratio2', color='brown')
#     #plt.fill_between(x, minmr2, maxmr2, alpha=0.3, color='brown')
# 
# if True:
#     plt.plot(x,ypft, lw=2, label='p_mr', color='cyan')
#     #plt.fill_between(x, minpft, maxpft, alpha=0.3, color='cyan')
#     plt.plot(x,yftmr, lw=2, label='ft_mr', color='black')
#     #plt.fill_between(x, minftmr, maxftmr, alpha=0.3, color='black')
#     plt.plot(x,ymrp, lw=2, label='mr_p', color='orange') 
#     #plt.fill_between(x, minmrp, maxmrp, alpha=0.3, color='orange')


plt.plot(np.ones_like(full_p[-1]),full_p[-1], lw=0, marker='.', color='green', alpha=0.3)
plt.plot(1, yp[-1], label='pressure', color='green', marker='o', ms=8)

plt.plot(2*np.ones_like(full_ft[-1]),full_ft[-1], lw=0, marker='.', color='blue', alpha=0.3)
plt.plot(2, yft[-1], label='firing times', color='blue', marker='o', ms=8)

plt.plot(3*np.ones_like(full_mr[-1]),full_mr[-1], lw=0, marker='.', color='red', alpha=0.3)
plt.plot(3, ymr[-1], label='mix ratio', color='red', marker='o', ms=8)

plt.plot(4*np.ones_like(full_p2[-1]),full_p2[-1], lw=0, marker='.', color='green', alpha=0.3)
plt.plot(4, yp2[-1], label='pressure$^2$', color='green', marker='o', ms=8)

plt.plot(5*np.ones_like(full_ft2[-1]),full_ft2[-1], lw=0, marker='.', color='blue', alpha=0.3)
plt.plot(5, yft2[-1], label='firing times$^2$', color='blue', marker='o', ms=8)

plt.plot(6*np.ones_like(full_mr2[-1]),full_mr2[-1], lw=0, marker='.', color='red', alpha=0.3)
plt.plot(6, ymr2[-1], label='mix ratio$^2$', color='red', marker='o', ms=8)

plt.plot(7*np.ones_like(full_pft[-1]),full_pft[-1], lw=0, marker='.', color='green', alpha=0.3)
plt.plot(7, yp2[-1], label='pft', color='green', marker='o', ms=8)

plt.plot(8*np.ones_like(full_mrp[-1]),full_mrp[-1]/100., lw=0, marker='.', color='blue', alpha=0.3)
plt.plot(8, ymrp[-1]/100., label='mrp', color='blue', marker='o', ms=8)

plt.plot(9*np.ones_like(full_ftmr[-1]),full_ftmr[-1], lw=0, marker='.', color='red', alpha=0.3)
plt.plot(9, yftmr[-1], label='mix ratio$^2$', color='red', marker='o', ms=8)

# 
# if True:
#     plt.plot(x,yp2, lw=2, label='pressure2', color='yellow')
#     #plt.fill_between(x, minp2, maxp2, alpha=0.3, color='yellow')
#     plt.plot(x,yft2, lw=2, label='firing times2', color='magenta')
#     #plt.fill_between(x, minft2, maxft2, alpha=0.3, color='magenta')
#     plt.plot(x,ymr2, lw=2, label='mixture ratio2', color='brown')
#     #plt.fill_between(x, minmr2, maxmr2, alpha=0.3, color='brown')
# 
# if True:
#     plt.plot(x,ypft, lw=2, label='p_mr', color='cyan')
#     #plt.fill_between(x, minpft, maxpft, alpha=0.3, color='cyan')
#     plt.plot(x,yftmr, lw=2, label='ft_mr', color='black')
#     #plt.fill_between(x, minftmr, maxftmr, alpha=0.3, color='black')
#     plt.plot(x,ymrp, lw=2, label='mr_p', color='orange') 
#     #plt.fill_between(x, minmrp, maxmrp, alpha=0.3, color='orange')

plt.xticks([1,2,3,4,5,6,7,8,9], ['pressure$\,$','firing times$\,$','mix ratio$\,$',
                                 'pressure$^2$','firing times$^2$','mix ratio$^2$',
                                 'p$\\times$ft','p $\\times$ mr / 100','mr$\\times$ft'],)
plt.xticks(rotation = 45)
plt.ylabel('failure rate factor = $h/{h_0}$')
#plt.legend(loc='best')
plt.xlim(0,10)
plt.ylim(0,5)
plt.title('covariates influence on failure rate')

from mayavi import mlab


def p_ft_func(p_array,ft_array):
    print p_array
    print ft_array
    surface = np.exp(p_array*p[-1] + ft_array*ft[-1] + p_array**2*p2[-1] + ft_array**2*ft2[-1] + p_array*ft_array*pft[-1])
    surface -= np.min(surface)
    return surface/np.max(np.abs(surface))

def p_mr_func(p_array,mr_array):
    print p_array
    print mr_array
    surface = np.exp(p_array*p[-1] + mr_array*mr[-1] + p_array**2*p2[-1] + mr_array**2*mr2[-1] + p_array*mr_array*pmr[-1])
    surface -= np.min(surface)
    return surface/np.max(np.abs(surface))


#p_grid, ft_grid = np.mgrid[0:130.:10, 0:600:10]
#mlab.surf(p_grid/130.,ft_grid/600.,p_ft_func)


p_grid, mr_grid = np.mgrid[0:130.:2, 0:7.:1]
mlab.surf(p_grid/130.,mr_grid/7.,p_mr_func)

mlab.show()


plt.figure()
ft_arr = np.linspace(0,600,100)
p_arr = np.linspace(0,130,100)
mr_arr = np.linspace(0,6,100)

plt.plot(ft_arr/ft_arr[-1], np.exp(ft_arr*ft[-1] + ft_arr**2*ft2[-1]))
plt.plot(p_arr/p_arr[-1], np.exp(p_arr*p[-1] + p_arr**2*p2[-1]))
plt.plot(mr_arr/mr_arr[-1], np.exp(mr_arr*mr[-1] + mr_arr**2*mr2[-1]))
plt.title('cavariates_effect')

plt.figure()
plt.title('baseline failure rate')
plt.xlabel('time [s]')
plt.ylabel('failure rate')
plt.plot(np.linspace(0,6000,500), m[-1]/s[-1]*(np.linspace(0,6000,500)/s[-1])**(m[-1]-1), color = 'black', lw=2)
plt.show()
