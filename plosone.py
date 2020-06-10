import jax.numpy as jnp
import numpy as np
from jax import jit, vmap, lax
from jax.ops import index,index_update
from jax.random import split,uniform,PRNGKey
from functools import partial
from matplotlib import pyplot as plt
import time
import numpy as np

# we set these and then we don't worry about them again
pfc_inputs = jnp.array( [ 3.0 , 0.0 ] )
w_pmc_d1 = 2.0
w_pmc_d2 = 2.0
d_gpe = 1.6
w_d2_gpe = 2.0
d_stn = 0.8
w_gpe_stn = 1.0
d_gpi = 0.2
w_d1_gpi = 1.4
w_stn_gpi = 1.6
d_pmc = 1.3
w_gpi_pmc = 1.8
w_pmc_pmc = 1.6
w_stn_gpe = 0.4
w_hd = 0.3
# learning rates; should be shape (3,1)
lrk =  jnp.array( [ 0.5,  -0.25, 0.0005 ] ).reshape((3,1)) # d1,d2,pmc
# forgetting rates; should be shape (3,1)
frk =  jnp.array( [ 0.02, 0.02, 0.0005 ] ).reshape((3,1)) # d1,d2,pmc

# configure simulation time step
dt = .15
nn = 5000


def ss( I ):
    # steady state function for neuronal populations
    # produces steady state firing rate given some input I.
    return jnp.tanh( jnp.clip(I, a_min =0) )


# time constants for bg populations in correct order for a single bg loop
tau = jnp.array([  15., 15., 20., 12.8, 15., 15., ])

def solve_bgloop( yy, rr, pfc, other_pmc,ww  ):
    # yy: state
    # rr: uniform noise over 0 , .1 ; same length as yy
    # pfc: activity of prefrontal cortex; array of length 2
    # other_pmc: activity of pmc from other bg loop
    # ww: we've already picked out the plastic weights for this bg loop
    # applies to pfc -> d1, pfc -> d2, pfc -> pmc
    
    # rename variables for sanity
    msnd1 = yy[0]
    msnd2 = yy[1]
    gpe   = yy[2]
    stn   = yy[3]
    gpi   = yy[4]
    pmc   = yy[5]

    I = jnp.array(
        [ jnp.dot( ww[0] , pfc ) + (w_pmc_d1 * pmc),
          jnp.dot( ww[1] , pfc ) + (w_pmc_d2 * pmc),
          d_gpe - (w_d2_gpe * msnd2) + (w_stn_gpe * stn),
          d_stn - (w_gpe_stn * gpe) + (w_hd * pmc),
          d_gpi - (w_d1_gpi * msnd1) + (w_stn_gpi * stn),
          d_pmc + jnp.dot( ww[2] , pfc ) - (w_gpi_pmc * gpi) - (w_pmc_pmc * other_pmc) ]
    )

    # we return the derivates
    return ( ( ss(I) - yy + rr) / tau )
    

def simulation_step_save_state( i, uk, w_pfc):
    # this method is used to produce a time series of a single simulation    
    # i: step in simulation. should run from 1 to length-1,
    # and i = 0 is the initial condition in uu
    # uk: this is the state that is carried over the loop
    # uk[0]: array for ODE state shape = (simulation length,14)
    # uk[1]: key for pseudorandom number generator. jax makes
    # us handle prng state ourselves
    # w_pfc: weights for plastic projections from pfc
    uu = uk[0] 
    vv = uu[i-1,:] 
    key = uk[1]
    key, subkey = split( key ) # jax makes us handle prng state ourselves
    rr = 0.1 * uniform( subkey, (12,) )
    # rename array for sanity
    pfc = vv[:2]
    goA = vv[2:2+6]
    goB = vv[2+6:]
    pfc_derivative = ( ss(pfc_inputs) - pfc ) / 15. # no  noise input here!
    goA_derivative = solve_bgloop( goA, rr[:6], pfc, goB[-1], w_pfc[0] )
    goB_derivative = solve_bgloop( goB, rr[6:] , pfc, goA[-1], w_pfc[1] )
    vv_derivative = jnp.concatenate( [ pfc_derivative ,
                                       goA_derivative ,
                                       goB_derivative ] )
    # do euler step for time i and update uu[i,:].
    # this is a jax.numpy array, so we use the index_update method
    # to update this row. this looks like it is copying uu, but
    # jax supposedly detects that the input array here is not getting
    # used again and does and in-place update instead of an array copy
    uu = index_update( uu, index[i,:], vv + vv_derivative*dt )
    # we are returning this list [uu,key] which is carried to the next
    # iteration of the loop: see uk in the method args
    return [uu,key] 

def simulation_step( i, vk, w_pfc):
    # this method performs a simulation without producing a
    # whole time series. only keeping last values for state.
    # i: step in simulation. should run from 1 to length-1,
    # we aren't actually using i in this version of the simulation
    # vk: this is the state that is carried over the loop
    # vk[0]: array for ODE state. shape = (14,)
    # vk[1]: key for pseudorandom number generator. jax makes
    # us handle prng state ourselves
    # w_pfc: weights for plastic projections from pfc
    
    vv = vk[0]
    key = vk[1]
    key, subkey = split( key ) # jax makes us handle prng state ourselves
    rr = 0.1 * uniform( subkey, (12,) )
    # rename array for sanity
    pfc = vv[:2]
    goA = vv[2:2+6]
    goB = vv[2+6:]
    pfc_derivative = ( ss(pfc_inputs) - pfc ) / 15. # no  noise input here
    goA_derivative = solve_bgloop( goA, rr[:6], pfc, goB[-1], w_pfc[0] )
    goB_derivative = solve_bgloop( goB, rr[6:] , pfc, goA[-1], w_pfc[1] )
    # TODO:
    # there might be a better way than this concatenate,
    # but the simulation is fast enough for me right now
    vv_derivative = jnp.concatenate( [ pfc_derivative ,
                                       goA_derivative ,
                                       goB_derivative ] )
    # do an euler step and update vv
    vv = index_update( vv, index[:], vv + vv_derivative*dt )
    return [vv,key]


def do_trial_for_figure( kw ):
    # we could break up this argument structure into something
    # more readable since this isn't getting used in a jax loop
    # kw: this is the trial state. 
    # kw[0]: key for prng
    # kw[1]: weights for plastic projections from pfc
    
    uu = jnp.zeros((nn,14))
    key = kw[0]
    w_pfc = kw[1]
    
    key, subkey = split( key ) # jax makes us handle prng state ourselves    
    uu = index_update( uu, index[0,:] , 0.1 * uniform(subkey,(7*2,) ))
    # set initial conditions before simulation
    # i've taken these details from plosone modeldb matlab code:
    # not sure how important are they
    uu = index_update( uu, index[0,4] , uu[0,4]+.6)
    uu = index_update( uu, index[0,4+6] , uu[0,4+6]+.6)
    uu = index_update( uu, index[0,0] , 0. )
    uu = index_update( uu, index[0,1] , 0. )
    uk = [uu,key]

    sim_step = partial( simulation_step_save_state, w_pfc = w_pfc )
    
    # for debugging purposes use this loop:
    #for i in range(uu.shape[0]-1):
    #    uk = sim_step(i+1,uk)
    
    # for performance use this "loop":
    uu,_ = lax.fori_loop( 1, uu.shape[0], sim_step, uk )
    return uu

def do_trial(kwr, reversal_learning):

    key = kwr[0]
    w_pfc = kwr[1]
    Re = kwr[2][0]
    key, subkey = split( key ) # jax makes us handle prng state ourselves    
    vv =  0.1 * uniform(subkey,(7*2,) )
    # set initial conditions before each simulation
    # i've taken these details from plosone modeldb matlab code:
    vv = index_update( vv, index[4] ,   vv[4]+.6   )
    vv = index_update( vv, index[4+6] , vv[4+6]+.6 )
    vv = index_update( vv, index[0] , 0. )
    vv = index_update( vv, index[1] , 0. )
    vk = [vv,key]

    sim_step = partial( simulation_step, w_pfc = w_pfc ) 
    
    # for debugging purposes use this loop:
    # (but don't call this do_trial methods hundreds of times)
    #for i in range(nn-1):
    #    vk = sim_step(i+1,vk)
    
    # for performance use this "loop":
    vv,key = lax.fori_loop( 1, nn, sim_step, vk )

    # rename variables for sanity
    pfc = vv[:2] # note this is of length two
    d1A = vv[2]
    d2A = vv[3]
    pmcA = vv[7]
    d1B = vv[2+6]
    d2B = vv[3+6]    
    pmcB = vv[7+6]
    pmc = [pmcA,pmcB]

    # these jax.lax.cond constructs replace some traditional
    # "if statement" condition logic blocks. this is done for
    # quick and easy jax.jit comaptibility. please check out
    # the jax documentation.

    # this picks the rewarded action
    rewardedAction, otherAction = lax.cond( reversal_learning,
                                            pmc,
                                            lambda x: [x[1],x[0]],
                                            pmc,
                                            lambda x: [x[0],x[1]] )

    # this determines reward for this trial
    R_trial = lax.cond( rewardedAction > otherAction + 0.1,
                        None,
                        lambda x: 1,
                        None,
                        lambda x: 0 )
    # reward prediction error:
    SNc = R_trial - Re

    # expected reward for next trial:
    a = 0.15
    Re_next = a * R_trial + (1 - a)*Re

    # weight updates:
    # the i,j,k notation below refers to a diagram that i drew and tacked
    # to my cork board. it should end up in git repo.
    # hopefully these comments explain the array
    # operations that we use to update all 12 weights with just few commands
    
    # i notation is ellided; that is the dimension of cues {#1,#2}
    # but we are instead just handling the vector pfc (shape = (2,))
    # j notation here indicates dimension of bg loops: {A,B}
    # k notation here indicates dimension of neuronal populations: {d1,d2,pmc}
    # qjk: array with population firing rates in each loop
    # sk: may modify qjk with SNc (which is reward prediction error)
    sk = jnp.array( [SNc,SNc,1] )
    qjk = jnp.array( [[d1A,d2A,pmcA],[d1B,d2B,pmcB]] )
    # sq: product of sk and qjk;
    # this should only modify d1,d2 msns by SNc; pmc is multiplied by 1
    # why? explanation:
    # pfc -> d1,d2 weights are modified by reward prediction error
    # pfc -> pmc weights are updated in a hebbian fashion
    sq = (sk * qjk).reshape((2,3,1))
    # we reshape this product from (2,3) to (2,3,1)
    # in preparation for weight update operations
    
    # weight update rule:    
    # pfc * sq 
    # (2,) * (2,3,1) -> (2,3,2), which is same shape as w_pfc
    # lrk: learning rate for each population (3,1)
    # lrk * ( the product of pfc and sq):
    # (3,1) * (2,3,2) -> (2,3,2)
    # frk: forgetting rate for each population (3,1)
    # frk * w_pfc:
    # (3,1) * (2,3,2) -> (2,3,2)
    dw_pfc = lrk * pfc * sq - frk * w_pfc
    
    # update weights; force new weights to be positive:
    w_pfc = jnp.clip( w_pfc + dw_pfc, a_min = 0)
    # w_pfc should be (2,3,2)

    return key, w_pfc, [Re_next,R_trial,SNc] , pmc

def session( key ):
    # this method is not a jax compatible method

    w_pfc = 0.001*uniform(key,(2,3,2)) # which loop, which target, which PFC
    Re = 1 # expected reward
    kwr = [key, w_pfc,[Re,0,0]] 
    # the zeros here in the third element are placeholders
    # because we are also using this list to return "actual reward"
    # and reward prediction error (see R_out below)

    # notice that these next three arrays are "regular numpy" arrays
    # once we descend into do_trial and its associated methods,
    # we will be dealing with "jax numpy" (jnp) arrays exclusively
    w_pfc1_out = np.zeros((2,3,500)) 
    R_out = np.zeros((3,500))
    pmc_out = np.zeros((2,500))

    # do_trial is jax.jit compatible. everything that it calls
    # is written to work with jax.jit.
    jit_do_trial = jit( do_trial  )
    
    tdiffs = [] # just used to benchmark peformance
    t0 = time.time()

    # now lets loop over the initial training trials
    # and save the interesting output
    for i in range( 200 ):
        t1 = time.time()
        *kwr,pmc = jit_do_trial( kwr, reversal_learning = False )
        w_pfc = kwr[1]
        w_pfc1_out[:,:,i] = w_pfc[:,:,0]
        R_out[:,i] = kwr[2]
        pmc_out[:,i] =pmc
        tdiff = time.time() - t1
        tdiffs.append(tdiff)

    # now we perform reversal learning trials
    # output is saved in same arrays as initial training trials
    for i in range( 200,500 ):
        t1 = time.time()        
        *kwr,pmc = jit_do_trial( kwr, reversal_learning = True )
        w_pfc = kwr[1]
        w_pfc1_out[:,:,i] = w_pfc[:,:,0]
        R_out[:,i] = kwr[2]
        pmc_out[:,i] =pmc
        tdiff = time.time() - t1
        tdiffs.append(tdiff)

    print('session duration',time.time() - t0 )
    print('median simulation duration',np.median(tdiffs)) # less than 4 ms on my laptop
    print('average simulation duration',np.mean(tdiffs))    
    return( w_pfc1_out, R_out,pmc_out )

def plot_all( uu ):
    # plots firing rates for bg looops
    w = 6
    lnt = nn*dt
    tt = np.arange(nn)*dt
    
    ax = plt.subplot(321)
    ax.plot(tt,uu[:,2])
    ax.plot(tt,uu[:,2+w])
    ax.set_xlim([0,lnt])
    ax.set_ylabel('d1 msn')
    #ax.set_ylim([0,1])
    
    ax = plt.subplot(322)
    ax.plot(tt,uu[:,3])
    ax.plot(tt,uu[:,3+w])
    ax.set_xlim([0,lnt])
    ax.set_ylabel('d2 msn')    
    #ax.set_ylim([0,1])
    
    ax = plt.subplot(323)
    ax.plot(tt,uu[:,4])
    ax.plot(tt,uu[:,4+w])
    ax.set_xlim([0,lnt])
    ax.set_ylabel('gpe')    
    #ax.set_ylim([0,1])
    
    ax = plt.subplot(324)
    ax.plot(tt,uu[:,5])
    ax.plot(tt,uu[:,5+w])
    ax.set_xlim([0,lnt])
    ax.set_ylabel('stn')
    #ax.set_ylim([0,1])
    
    ax = plt.subplot(325)
    ax.plot(tt,uu[:,6])
    ax.plot(tt,uu[:,6+w])
    ax.set_xlim([0,lnt])
    ax.set_ylabel('gpi')    
    #ax.set_ylim([0,1])
    
    ax = plt.subplot(326)
    ax.plot(tt,uu[:,7])
    ax.plot(tt,uu[:,7+w])
    ax.set_xlim([0,lnt])
    ax.set_ylabel('pmc')        
    #ax.set_ylim([0,1])
    
    plt.show()



if __name__ == '__main__':

    # this set of simulations produces the synaptic weight plots
    # for initial learning and reversal learning for the
    # healthy BG from plosone paper as in Fig4 and Fig6
    key = PRNGKey( time.time_ns() )
    w,r,p = session( key )
    # w: weights for pfc1 projections.
    # shape is (2,3,500) ; ( {A or B?}, {d1, d2, or pmc?}, {trial number})
    # r: expected reward, actual reward, reward prediction error
    # shape is (3,500)
    # p: value of pmcA,pmcB activity at the end of each trial
    # shape is (2,500)
    
    # synaptic weighths plot    
    ax = plt.subplot(3,1,1)
    # pmc activity at end of each trial
    for x in p:
        ax.plot(x)
    ax.set_ylim([0,1])
    ax.set_xlim([0,500])
    ax = plt.subplot(3,1,2)
    # w_pfc: pfc1 -> d1,d2; loops a,b
    for x in w[:,:2,:]:
        for y in x:
            ax.plot(y)
    ax.set_ylim([0,2])
    ax.set_xlim([0,500])
    ax = plt.subplot(3,1,3)
    # w_pfc: pfc1 -> pmc; loops a,b
    for x in w[:,2,:]:
        ax.plot(x)
    ax.set_ylim([0,.15])
    ax.set_xlim([0,500])
    plt.show()

    # reward feebdack plots
    ax = plt.subplot(2,1,1)
    ax.plot( r[1] ) # expected reward 
    # maybe expected reward is shifted by one trial compared to plosone.
    # should it be expected reward at the start of the trial?
    # or expected reward for next trial?
    ax.plot( r[0] ) # actual reward
    ax.set_xlim([0,500])
    
    ax = plt.subplot(2,1,2)
    ax.plot( [0,500],[0,0],'--', color = [0,0,0])    
    ax.plot( r[2] ) # reward prediction error
    ax.set_ylim([-1,1])
    ax.set_xlim([0,500])    
    plt.show()


    
    # this simulation produces Healthy BG population activity
    # from the plosone paper as in Fig4 and Fig6
    key = PRNGKey( time.time_ns() )
    w_pfc = 0.01*uniform(key,(2,3,2))
    w_pfc = index_update( w_pfc, index[0,0,0], 0.7 )
    w_pfc = index_update( w_pfc, index[1,1,0], 0.7 )
    uu = jnp.zeros((nn,14))
    uu = do_trial_for_figure( [key, w_pfc] ) 
    plot_all(uu)
