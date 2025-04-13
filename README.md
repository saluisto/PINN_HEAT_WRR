# PINN_HEAT_WRR
Physics-Informed Neural Network for the inverse estimation of submarine groundwater discharge by using heat as a tracer.

## Aim of the project
The code is designed to perform inverse estimation of groundwater fluxes by leveraging measured temperature profiles in the subsurface. This is achieved through the inverse solution of the heat conduction/convection equation . The inverse simulation employs a Physics-Informed Neural Network (PINN) approach, implemented using Python, TensorFlow, and the deepxde library.

## Why is the project useful?
Traditionally, inverse simulations rely on numerical or analytical solutions of the heat transport equation. However, these methods exhibit insensitivity to high-frequency flux variations, often encountered in dynamic environments such as high-energy beach faces or tidal systems. The presented proof-of-concept PINN framework overcomes this limitation, demonstrating a notable capability to accurately quantify groundwater fluxes in such systems. Its resilience to high-frequency variations suggests a promising potential for reliable applications in these dynamic environments.

## How can the community contribute?
The PINN network is currently a work in progress and has only been applied to a relatively limited dataset. The learning process of the PINN is not yet optimized, and convergence remains slow. The primary objective for the future is to enhance the efficiency of the PINN, enabling its application to extended and finely resolved datasets. Community support and collaboration are highly appreciated in achieving this goal. Contributions and insights from the community would greatly aid in refining the PINN's performance for broader and more complex datasets.

##Where can the community can get assitance?
sven.frei@wur.nl
***
# Data
The PINN_heat uses a synthetic/observed dataset of observed temperatures for a different depths for the inverse solution of the advective flow velocities:

1) SUTRA dataset on meso-tidal cycles (SUTRA_meso.csv) published in:

  LeRoux, N. K., Kurylyk, B. L., Briggs, M. A., Irvine, D. J., Tamborski, J. J., & Bense, V. F. 547 
  (2021). Using Heat to Trace Vertical Water Fluxes in Sediment Experiencing 548 
  Concurrent Tidal Pumping and Groundwater Discharge. Water Resources Research, 549 
  57(2), e2020WR027904. https://doi.org/10.1029/2020WR027904

2)



3) Unpublished temperature measuremnts from tidal creek system on Texel, Netherlands (K2)
