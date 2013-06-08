################################################################################
#
# Program:      2D BEC simulation with gravitational potential 
#
# Author:       Nathanael Lampe, School of Physics, Monash University
#
# Created:      December 4, 2012 (Forked from twodtrap.py)
#
# Changelog:    twodtrap.py
#               Aug 29, 2012: Created
#               Oct 11, 2012: Working, changelog begun
#               Oct 11, 2012: Implementation of rotation into Hamiltonian
#               twod.py
#               Dec 04, 2012: Created from twodtrap.py
#               Dec 12, 2012: Implemented scheme to check cnvergence against the
#                             chemical potential
#               Jan 24, 2012: The program works in accordance with the virial 
#                             theorem, and TF approximation, Has now been forked
#                             to rotate.py, where rotation will be implemented.
#               Jan 30, 2013: Saving functionality REMOVED!
#                             NOW ONLY CONTAINS BOSE CLASS AND RELATED FUNCTIONS
#
#               Jun 08, 2013: 
#               
#
# Purpose:      Provide a class to evaluate the stability of a 
#               gravitationally bound BEC under varying trap strengths
#               
################################################################################

#Imports:
from __future__ import division

import numpy as np
import scipy.fftpack as ff
import scipy.linalg as la
import matplotlib.pyplot as plt

from scipy.sparse import spdiags
from scipy.special import j0, j1, jn_zeros

################################################################################

def k_vector( n, extent ):
  '''Usual FFT format k vector (as in Numerical Recipes)'''
  delta = float(extent) / float(n-1)
  k_pos = np.arange(      0, n/2+1, dtype = float)
  k_neg = np.arange( -n/2+1,     0, dtype = float)
  return (2*np.pi/n/delta) * np.concatenate( (k_pos,k_neg) )

################################################################################

def k2dimen(a,b,npt):
  k_vec = k_vector(npt, b - a )
  KX, KY = np.meshgrid ( k_vec, k_vec )
  k2squared = KX ** 2. + KY ** 2.
  k2 = np.sqrt( k2squared )
  
  #k_vec = k_vector(npt, b - a )
  #k2d = np.zeros([npt,npt])
  #kf2 = np.zeros([npt,npt])
  #for ii in range(0,npt):
    #for jj in range(0,npt):
      #kf2[ii,jj] = k_vec[ii]**2+k_vec[jj]**2
      #k2d[ii,jj] = np.sqrt(kf2[ii,jj])
  k2[ npt/2:npt, 0:npt/2   ] = -k2[ npt/2:npt, 0:npt/2   ]
  k2[ 0:npt/2  , npt/2:npt ] = -k2[ 0:npt/2  , npt/2:npt ]
  return k2squared, k2

################################################################################

def harm_trap( x, y, P ):
  X, Y = np.meshgrid(x,y)
  return 0.5 * P * ( X**2 + Y**2 )

################################################################################
  
def kernel( x    , 
            y    ,
            npt  ):
  '''
  Define the grid 1/abs(r)
  '''
  ker = np.zeros([npt,npt])
  for ii in range(0,npt):
    for jj in range(0,npt):
      ker[ii,jj] = abs(1./np.sqrt(x[ii]**2.+y[jj]**2.))
  return ker

################################################################################
  
class Bose:
  '''
  Establish the Bose class
  The Bose class defines an item that contains most relevant arrays
  that are passed to functions. Many of these are to optimise the routine
  '''
  def __init__( self, a, b, npt, init, g, G, P, dt, **kwargs ):
    self.xmin = a
    self.xmax = b
    self.npt  = npt
    self.P    = P
    self.g    = g
    self.G    = G
    self.dt   = dt
    
    self.x  = np.linspace( self.xmin, self.xmax, npt )
    self.dx = abs( self.x[1] - self.x[0] )
    self.y  = self.x
    self.dy = abs( self.y[1] - self.y[0] )
    self.X, self.Y = np.meshgrid(self.x, self.y)
    
    self.psi = init(self.x, self.y, self.g, self.G, **kwargs)
    self.psi = self.psi / np.sqrt(sum(sum(abs(self.psi)**2.))*self.dx*self.dy)
    #This ensures the wavefunction is normalised
    
    self.ksquare , self.k = k2dimen   (self.xmin, self.xmax, self.npt)
    self.V                = harm_trap (self.x, self.y, self.P)
    self.ker              = kernel    (self.x, self.y, self.npt)
    
    self.expksquare       = np.exp    (-0.5j*self.dt*self.ksquare)
    
    self.log              = np.log(np.sqrt(self.X ** 2. + self.Y ** 2.))
    
    
  #-----------------------------------------------------------------------------
  
  def gravity(self):
    '''
    Evaluate the gravitational field, with a call to Bose.gravity()
    Gravitaional field is the convolution of the density and the log of distance
    '''
    den = abs(self.psi)**2.  #calculate the probability density
    
    #return the convolution, after multiplying by scaled gravity and 
    #correcting for grid scaling (due to 2 forward FFTs, and only one inverse
    return self.G*self.dx*self.dy*(ff.fftshift(ff.ifft2(ff.fft2(ff.fftshift(den)
      )*abs(ff.fft2(ff.fftshift(-self.log))))))
  
  #-----------------------------------------------------------------------------  
  
  def angular(self):
    '''
    return the angular momentum operator value at each point in the grid
    L = -i ( x * d/dy - y * d/dx ) * psi
    We assume a square grid centred on  (x,y) = (0,0)
    '''
    kx = k_vector( self.npt, self.xmax - self.xmin )
    ky = kx
    
    KX, KY = np.meshgrid( kx, ky )
    X ,  Y = np.meshgrid( self.x, self.y )    
    fpsi   = ff.fft2( ff.fftshift( self.psi ) )
    
    
    return self.rot *1j * ( X * ff.fftshift( ff.ifft2( -1j * KY * fpsi) ) -
                            Y * ff.fftshift( ff.ifft2( -1j * KX * fpsi) )
                          ) 
  #-----------------------------------------------------------------------------
  
  def step2(self):
    '''
    Perform a second order timestep
    '''
    
    V2 = np.exp( -1j * self.dt / 2. * (self.V - self.gravity() + 
                                       self.g * abs( self.psi ) ** 2
                                       ) 
                )
                   
    return V2 * ff.fftshift( ff.ifft2(self.expksquare * 
                                    ff.fft2( ff.fftshift( V2 * self.psi ) )
                                    )
                           )
  
  #-----------------------------------------------------------------------------
  
  def step4(self):
    '''
    Perform a 4th order timestep
    '''
    
    def order2(c):
      Vc = np.exp( -1j * c * self.dt / 2. * 
                    ( self.V - self.gravity() + 
                      self.g * abs( self.psi ) ** 2
                    )
                  )
      Tc = self.expksquare ** c
      return Vc, Tc
    
    p = 1/(4.-4.**(1/3.))
    q = 1 - 4 * p
    
    Vp,Tp = order2( p )
    Vq,Tq = order2( q )
    
    return Vp * ff.fftshift( ff.ifft2( Tp * ff.fft2( ff.fftshift( Vp ** 2 * 
                ff.fftshift( ff.ifft2( Tp * ff.fft2( ff.fftshift( Vp * Vq *
                ff.fftshift( ff.ifft2( Tq * ff.fft2( ff.fftshift( Vq * Vp * 
                ff.fftshift( ff.ifft2( Tp * ff.fft2( ff.fftshift( Vp ** 2 *  
                ff.fftshift( ff.ifft2( Tp * ff.fft2( ff.fftshift( Vp * self.psi 
                ) ) ) )
                ) ) ) )
                ) ) ) )
                ) ) ) )
                ) ) ) )
    
  #-----------------------------------------------------------------------------
  
  def wickon(self):
    '''
    simple function to ensure imaginary time propagation
    '''
    self.dt               = -1j * abs(self.dt)
    self.expksquare       = np.exp( -0.5j * self.dt * self.ksquare )
  #-----------------------------------------------------------------------------
  
  def wickoff(self):
    '''
    simple function to ensure real time propagation
    '''
    self.dt               = abs(self.dt)
    self.expksquare       = np.exp( -0.5j * self.dt * self.ksquare )
  #-----------------------------------------------------------------------------
  
  def energies( self            ,
                verbose = False ):
    '''
    energies(self, verbose = False)
    bec is an instance of the Bose class
    Returns the list of energies enList = [ Ev, Ei, Eg, Ekin ]
    This is the harmonic, interaction, gravitational and kinetic energies
    '''
    
    
    Ev = sum( sum( self.psi.conjugate() * self.V * self.psi )
            ) * self.dx * self.dy
    
    Ei = sum( sum( self.psi.conjugate() * 0.5 * self.g * abs(self.psi) **2. *
                   self.psi ) ) * self.dx * self.dy
    
    Eg = sum( sum( self.psi.conjugate() * -1. * self.gravity() * self.psi )
            ) * self.dx * self.dy
    
    Ekin = sum( 
               sum(
        self.psi.conjugate() * ff.fftshift( ff.ifft2( 0.5 * self.ksquare * 
                                            ff.fft2( ff.fftshift( self.psi ) ) ) 
                                          )
                   )
              ) * self.dx * self.dy
    
    enList = [ Ev, Ei, Eg, Ekin ]
    
    
    
    if verbose == True: #print out the energies
      #Calculate gravitational field Laplacian
      gf = -1. * self.gravity() #self.gravity() is +ve
      glpg = ( np.roll( gf, 1, axis = 0 )
             + np.roll( gf,-1, axis = 0 )
             + np.roll( gf, 1, axis = 1 )
             + np.roll( gf,-1, axis = 1 )
             - 4 * gf ) / self.dx ** 2.
      glpd = 2. * np.pi * self.G * abs(self.psi) ** 2.
      
      ##diagnostic plots
      #gravfig = plt.figure()
      #gravax = gravfig.add_subplot(111)
      #gravax.plot(self.x, glpg[ self.npt//2, : ], label = 'Fourier Laplace' )
      #gravax.plot(self.x, glpd[ self.npt//2, : ], label = 'Density Laplace' )
      #gravlgd = plt.legend(loc='upper right')
      #plt.show()
      
      print 'Harmonic PE         = ', np.real( enList[0] )
      print 'Interaction PE      = ', np.real( enList[1] )
      print 'Gravitational PE    = ', np.real( enList[2] )
      print 'Potential Energy    = ', np.real( sum( enList[0:3] ) )
      print 'Kinetic Energy      = ', np.real( enList[3] )
      print 'Total Energy        = ', np.real( sum( enList ) )
      print 'normalisation       = ', np.real( sum( sum( abs( self.psi ) ** 2. ) 
                                                  ) * self.dx * self.dy )
      print 'Chemical Potential  = ', np.real( enList[3] + enList[0] + 
                                               2 * enList[1] + enList[2] )
      print 'Ek - Ev + Ei - G/4  = ', np.real( enList[3] - enList[0] + enList[1]
                                             - self.G/4. )
      print 'Laplace Eq check    = ', np.real( sum( sum( glpg - glpd ) ) )
    return enList
  #-----------------------------------------------------------------------------
  
  def TFError( self,
               verbose = False):
    '''
    TFError( self, verbose = False):
    A function to evaluate the deviation of the solution from the Thomas-Fermi
    approximation.
    Setting verbose to true will produce plots of the wavefunction and TF 
    approximation
    '''
    
    enList = self.energies( verbose = False )
    #mu    = <K>       + <Vext>    + 2 * <Vsi>     + 0.5 * <Vgrav>
    fineX = np.arange(self.xmin , self.xmax, (self.xmax - self.xmin) / 
                                             ( 1e3 * self.npt ), dtype = float )
    fineY = fineX
    X, Y  = np.meshgrid( self.x, self.y )
    Rsq   = X ** 2. + Y ** 2.
    R     = np.sqrt( Rsq )
    
    if self.g != 0 and self.G == 0. and self.P != 0.:
      #Harmonic case
      chmpot = enList[3] + enList[0] + 2 * enList[1]
      
      r0sq   = 2 * chmpot / self.P
      tfsol  = (chmpot / self.g - Rsq * self.P / ( 2 * self.g ) ) * ( 
                                          map( lambda rsq: rsq - r0sq < 0, Rsq )
                                               )
      tferror = np.real( sum( sum( tfsol - abs( self.psi ) ** 2. ) ) )
      print 'Thomas-Fermi Error = ', tferror
      print 'TF Error per cell  = ', tferror / self.npt ** 2.
      
      if verbose == True:
        #diagnostic plots
        
        tfx  = (chmpot / self.g - fineX ** 2. * self.P / ( 2 * self.g ) ) * ( 
                                     map( lambda fineX: fineX **2. - r0sq < 0,
                                          fineX )                           )
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.plot(fineX, tfx, label = 'analytic solution')
        
        ax.plot(self.x, abs( self.psi[ self.npt//2, : ] ) ** 2.,
                                                  label = 'numerical solution' )
        
        lgd = plt.legend( loc = 'upper right' )
        plt.show()
      
      
    elif self.G != 0. and self.g != 0. and self.P == 0.:
      #Gravitational case
      bj0z1   = jn_zeros( 0, 1 ) #First zero of zeroth order besselj
      scaling = np.sqrt( 2 * np.pi * self.G / self.g  )
      gr0     = bj0z1 / scaling
      Rprime  = R * scaling
      
      gtfsol = j0( Rprime ) * np.array( [ map( int,ii ) for ii in map( 
                                         lambda rad: rad <= gr0, R ) ] )

      
      gtfsol *= scaling ** 2. / ( 2 * np.pi * j1( bj0z1 ) * bj0z1 ) #(abs(self.psi) ** 2.).max()
      
      #gtfsol = 1. / ( 4. * gr0 ** 2. * ( 1 - 2 / np.pi ) ) * np.cos(
        #np.pi * R / ( 2. * gr0 ) ) * map( lambda rsq: rsq - gr0 ** 2. < 0, Rsq )
      gtferror = np.real( sum( sum( gtfsol - abs( self.psi ) ** 2. ) ) ) 
      
      print 'Grav. TF Error     = ', gtferror
      print 'GTF Error per cell = ', gtferror / self.npt ** 2.
      print 'Analytic norm      = ', sum( sum( gtfsol ) ) * self.dx * self.dy
      
      if verbose == True:
        #diagnostic energies
        gtfwf = np.sqrt( gtfsol )
        
        Ev = 0.
        
        Ei = sum( sum( 0.5 * self.g * abs(gtfwf) ** 4. ) 
                ) * self.dx * self.dy
        
        GField = self.G * self.dx * self.dy * (
                 ff.fftshift( ff.ifft2( ff.fft2( ff.fftshift( gtfsol ) ) * 
                       abs( ff.fft2( ff.fftshift( -self.log ) ) ) ) 
                            )                 )
        Eg = sum( sum( -1. * GField * gtfsol ) ) * self.dx * self.dy
        
        Ekin = sum( sum( gtfwf.conjugate() * 
                         ff.fftshift( ff.ifft2( 0.5 * self.ksquare * 
                         ff.fft2( ff.fftshift( gtfwf ) ) ) 
                                     )
                       )
                  ) * self.dx * self.dy
        
        TFList = [ Ev, Ei, Eg, Ekin ]
        
        #glpg = - 1. * ( np.roll( GField, 1, axis = 0 ) #GField is +ve
                      #+ np.roll( GField,-1, axis = 0 )
                      #+ np.roll( GField, 1, axis = 1 )
                      #+ np.roll( GField,-1, axis = 1 )
                      #- 4 * GField ) / self.dx ** 2.
        #glpd = 2 * np.pi * self.G * gtfsol 
        
        print '\nTF solution Energies\n'
        print 'Harmonic PE         = ', np.real( TFList[0] )
        print 'Interaction PE      = ', np.real( TFList[1] )
        print 'Gravitational PE    = ', np.real( TFList[2] )
        print 'Potential Energy    = ', np.real( sum( TFList[0:3] ) )
        print 'Kinetic Energy      = ', np.real( TFList[3] )
        print 'Total Energy        = ', np.real( sum( TFList ) )
        print 'Chemical Potential  = ', np.real( TFList[3] + TFList[0] + 
                                                2 * TFList[1] + TFList[2] )
        print 'Ek - Ev + Ei - G/4  = ', np.real( TFList[3] - TFList[0] +
                                                  TFList[1] - self.G / 4. )
        
        #diagnostic plots
        fineXS = fineX * scaling
        gtx = j0( fineXS ) * np.where( abs(fineX) < gr0,
                              np.ones( len(fineX) ), np.zeros( len(fineX) ) )
        gtx *= scaling ** 2. / ( 2 * np.pi * j1( bj0z1 ) * bj0z1 )
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.plot(fineX, gtx, label = 'analytic solution')
        ax.plot(self.x, abs( self.psi[ self.npt//2, : ] ) ** 2.,
                                                  label = 'numerical solution' )
        
        lgd = plt.legend( loc = 'upper right' )
        plt.show()
      
    else:
      print 'A TF approximation for this scenario has not yet been implemented'
      print 'Sorry about that...'
      
  #-----------------------------------------------------------------------------
  
  def virial( self            ,
              verbose = False ):
    '''
    def converge( self            ,
                  verbose = False ):
    A routine to check if the ground state has been reached.
    Returns the virial
    Verbose will print the virial.
    virial = 'Ek - Ev + Ei - G/4'
    '''
    enList = self.energies( verbose = False )
    virial = np.real( enList[3] - enList[0] + enList[1] - self.G/4. )
    if verbose == True: print 'Ek - Ev + Ei - G/4 = ', virial
    return virial