/**
 * @file
 * This file is part of SWE.
 *
 * @author Alexander Breuer (breuera AT in.tum.de, http://www5.in.tum.de/wiki/index.php/Dipl.-Math._Alexander_Breuer)
 * @author Sebastian Rettenberger (rettenbs AT in.tum.de, http://www5.in.tum.de/wiki/index.php/Sebastian_Rettenberger,_M.Sc.)
 * @author Michael Bader (bader AT in.tum.de, http://www5.in.tum.de/wiki/index.php/Michael_Bader)
 *
 * @section LICENSE
 *
 * SWE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SWE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SWE.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 * @section DESCRIPTION
 *
 * SWE_Block, which uses solvers in the wave propagation formulation.
 */

#include "SWE_WaveAccumulationBlock.hh"

#include <cassert>
#include <string>
#include <limits>
#include <immintrin.h>

//#define USE_SCALASCA

#ifdef USE_SCALASCA
#include "epik_user.h"
#endif

#ifdef LOOP_OPENMP
#include <omp.h>
#endif

#define BLOCK_SIZE 4
/**
 * Constructor of a SWE_WaveAccumulationBlock.
 *
 * Allocates the variables for the simulation:
 *   unknowns h,hu,hv,b are defined on grid indices [0,..,nx+1]*[0,..,ny+1] (-> Abstract class SWE_Block)
 *     -> computational domain is [1,..,nx]*[1,..,ny]
 *     -> plus ghost cell layer
 *
 * Similar, all net-updates are defined as cell-local variables with indices [0,..,nx+1]*[0,..,ny+1], 
 * however, only values on [1,..,nx]*[1,..,ny] are used (i.e., ghost layers are not accessed).
 * Net updates are intended to hold the accumulated(!) net updates computed on the edges.
 *
 */
SWE_WaveAccumulationBlock::SWE_WaveAccumulationBlock(
		int l_nx, int l_ny,
		float l_dx, float l_dy):
  SWE_Block(l_nx, l_ny, l_dx, l_dy),
  hNetUpdates (nx+2, ny+2),
  huNetUpdates(nx+2, ny+2),
  hvNetUpdates(nx+2, ny+2)
{}

/**
 * Compute net updates for the block.
 * The member variable #maxTimestep will be updated with the 
 * maximum allowed time step size
 */
void SWE_WaveAccumulationBlock::computeNumericalFluxes() {

#ifdef USE_SCALASCA
	EPIK_FUNC_START();
#endif

	float dx_inv = 1.0f/dx;
	float dy_inv = 1.0f/dy;

	//maximum (linearized) wave speed within one iteration
	float maxWaveSpeed = (float) 0.;

	// compute the net-updates for the vertical edges

	int ny_end = ny+1;	// compiler might refuse to vectorize j-loop without this ...
#ifdef LOOP_OPENMP
#pragma omp parallel
{
#endif // LOOP_OPENMP
	// thread-local maximum wave speed:
	float l_maxWaveSpeed = (float) 0.;

	
#ifdef VECTORIZE
	__assume_aligned(&h, 64);
	__assume_aligned(&hu, 64);
	__assume_aligned(&b, 64);
	__assume_aligned(&hNetUpdates, 64);
	__assume_aligned(&huNetUpdates, 64);

	__declspec(align(64)) float arr_hNetUpLeft[BLOCK_SIZE]; 
	__declspec(align(64)) float arr_hNetUpRight[BLOCK_SIZE]; 
	__declspec(align(64)) float arr_huNetUpLeft[BLOCK_SIZE]; 
	__declspec(align(64)) float arr_huNetUpRight[BLOCK_SIZE]; 
	__declspec(align(64)) float arr_maxEdgeSpeedVerti[BLOCK_SIZE];
#endif

	//std::cout << "ny_end: " << ny_end << std::endl;
	
#ifdef LOOP_OPENMP
	// Use OpenMP for the outer loop
	#pragma omp for
#endif // LOOP_OPENMP
	for(int i = 1; i < nx+2; i++) {

	#ifdef VECTORIZE // Vectorize the inner loop
		// BLOCK_SIZE has to be a divisor of the grid-size
		for (int iBlock = 1; iBlock < ny_end; iBlock += BLOCK_SIZE)
		{
			wavePropagationSolver.computeNetUpdates( h[i-1][iBlock:BLOCK_SIZE], h[i][iBlock:BLOCK_SIZE],
												   hu[i-1][iBlock:BLOCK_SIZE], hu[i][iBlock:BLOCK_SIZE],
												   b[i-1][iBlock:BLOCK_SIZE], b[i][iBlock:BLOCK_SIZE],
												   arr_hNetUpLeft[:], arr_hNetUpRight[:],
												   arr_huNetUpLeft[:], arr_huNetUpRight[:],
												   arr_maxEdgeSpeedVerti[:] );
				
			// accumulate net updates to cell-wise net updates for h and hu
			hNetUpdates[i-1][iBlock:BLOCK_SIZE]  += dx_inv * arr_hNetUpLeft[:];
			huNetUpdates[i-1][iBlock:BLOCK_SIZE] += dx_inv * arr_huNetUpLeft[:];
			hNetUpdates[i][iBlock:BLOCK_SIZE]    += dx_inv * arr_hNetUpRight[:];
			huNetUpdates[i][iBlock:BLOCK_SIZE]   += dx_inv * arr_huNetUpRight[:];
			
			#ifdef LOOP_OPENMP
				//update the thread-local maximum wave speed
				l_maxWaveSpeed = std::max(l_maxWaveSpeed, __sec_reduce_max(arr_maxEdgeSpeedVerti[:]));
			#else // LOOP_OPENMP
				//update the maximum wave speed
				maxWaveSpeed = std::max(maxWaveSpeed, __sec_reduce_max(arr_maxEdgeSpeedVerti[:]));
			#endif // LOOP_OPENMP
		}	
			
		/* Code producing strange differences in output files
		for(int j = 1; j < ny_end; j++) {

			float hNetUpLeft, hNetUpRight;
			float huNetUpLeft, huNetUpRight;

			wavePropagationSolver.computeNetUpdates( h[i-1][j], h[i][j],
											   hu[i-1][j], hu[i][j],
											   b[i-1][j], b[i][j],
											   arr_hNetUpLeft[j-1], arr_hNetUpRight[j-1],
											   arr_huNetUpLeft[j-1], arr_huNetUpRight[j-1],
											   arr_maxEdgeSpeedVerti[j-1] );
											   
			// accumulate net updates to cell-wise net updates for h and hu
			//hNetUpdates[i-1][j]  += dx_inv * hNetUpLeft;
			//huNetUpdates[i-1][j] += dx_inv * huNetUpLeft;
			//hNetUpdates[0][j]    += dx_inv * arr_hNetUpRight[j-1];
			//huNetUpdates[i][j]   += dx_inv * arr_huNetUpRight[j-1];
		}
		
		huNetUpdates[i][1:ny_end - 1]   += dx_inv * arr_huNetUpRight[:];
		
		hNetUpdates[i-1][1:ny_end - 1]  += dx_inv * arr_hNetUpLeft[:];
		hNetUpdates[i][1:ny_end - 1]    += dx_inv * arr_hNetUpRight[:];		
		huNetUpdates[i-1][1:ny_end - 1] += dx_inv * arr_huNetUpLeft[:];
		
		#ifdef LOOP_OPENMP
			//update the thread-local maximum wave speed
			l_maxWaveSpeed = std::max(l_maxWaveSpeed, __sec_reduce_max(arr_maxEdgeSpeedVerti[:]));
		#else // LOOP_OPENMP
			//update the maximum wave speed
			maxWaveSpeed = std::max(maxWaveSpeed, __sec_reduce_max(arr_maxEdgeSpeedVerti[:]));
		#endif // LOOP_OPENMP
		*/
	#else
		for(int j = 1; j < ny_end; j++) {

			float maxEdgeSpeed;
			float hNetUpLeft, hNetUpRight;
			float huNetUpLeft, huNetUpRight;

			wavePropagationSolver.computeNetUpdates( h[i-1][j], h[i][j],
											   hu[i-1][j], hu[i][j],
											   b[i-1][j], b[i][j],
											   hNetUpLeft, hNetUpRight,
											   huNetUpLeft, huNetUpRight,
											   maxEdgeSpeed );

			// accumulate net updates to cell-wise net updates for h and hu
			hNetUpdates[i-1][j]  += dx_inv * hNetUpLeft;
			huNetUpdates[i-1][j] += dx_inv * huNetUpLeft;
			hNetUpdates[i][j]    += dx_inv * hNetUpRight;
			huNetUpdates[i][j]   += dx_inv * huNetUpRight;

			#ifdef LOOP_OPENMP
				//update the thread-local maximum wave speed
				l_maxWaveSpeed = std::max(l_maxWaveSpeed, maxEdgeSpeed);
			#else // LOOP_OPENMP
				//update the maximum wave speed
				maxWaveSpeed = std::max(maxWaveSpeed, maxEdgeSpeed);
			#endif // LOOP_OPENMP
		}
	#endif // VECTORIZE
	}

	// compute the net-updates for the horizontal edges

	ny_end = ny+2;	// compiler refused to vectorize j-loop without this ...
#ifdef VECTORIZE // Vectorize the inner loop	
	float arr_hNetUpDown[ny_end - 1]; 
	float arr_hNetUpUpwards[ny_end - 1]; 
	float arr_hvNetUpDown[ny_end - 1]; 
	float arr_hvNetUpUpwards[ny_end - 1]; 
	float arr_maxEdgeSpeedHori[ny_end - 1];
#endif

#ifdef LOOP_OPENMP // Use OpenMP for the outer loop
	#pragma omp for
#endif // LOOP_OPENMP

	for(int i = 1; i < nx+1; i++) {

	#ifdef VECTORIZE // Vectorize the inner loop	
	/*	Vectorized Code with Intel Array Notation */
		wavePropagationSolver.computeNetUpdates( h[i][0:ny_end - 1], h[i][1:ny_end - 1],
										   hv[i][0:ny_end - 1], hv[i][1:ny_end - 1],
										   b[i][0:ny_end - 1], b[i][1:ny_end - 1],
										   arr_hNetUpDown[:], arr_hNetUpUpwards[:],
										   arr_hvNetUpDown[:], arr_hvNetUpUpwards[:],
										   arr_maxEdgeSpeedHori[:] );
		// accumulate net updates to cell-wise net updates for h and hu
		hNetUpdates[i][0:ny_end - 1]  += dy_inv * arr_hNetUpDown[:];
		hvNetUpdates[i][0:ny_end - 1] += dy_inv * arr_hvNetUpDown[:];
		hNetUpdates[i][1:ny_end - 1]    += dy_inv * arr_hNetUpUpwards[:];
		hvNetUpdates[i][1:ny_end - 1]   += dy_inv * arr_hvNetUpUpwards[:];	
		
		#ifdef LOOP_OPENMP
			//update the thread-local maximum wave speed
			l_maxWaveSpeed = std::max(l_maxWaveSpeed, __sec_reduce_max(arr_maxEdgeSpeedHori[:]));
		#else // LOOP_OPENMP
			//update the maximum wave speed
			maxWaveSpeed = std::max(maxWaveSpeed, __sec_reduce_max(arr_maxEdgeSpeedHori[:]));
		#endif // LOOP_OPENMP		
	
	/* Not vectorized Code 
		for(int j = 1; j < ny_end; j++) {
			float maxEdgeSpeed;
			float hNetUpDow, hNetUpUpw;
			float hvNetUpDow, hvNetUpUpw;

			wavePropagationSolver.computeNetUpdates( h[i][j-1], h[i][j],
											   hv[i][j-1], hv[i][j],
											   b[i][j-1], b[i][j],
											   hNetUpDow, hNetUpUpw,
											   hvNetUpDow, hvNetUpUpw,
											   maxEdgeSpeed );

			// accumulate net updates to cell-wise net updates for h and hu
			hNetUpdates[i][j-1]  += dy_inv * hNetUpDow;
			hvNetUpdates[i][j-1] += dy_inv * hvNetUpDow;
			hNetUpdates[i][j]    += dy_inv * hNetUpUpw;
			hvNetUpdates[i][j]   += dy_inv * hvNetUpUpw;

			#ifdef LOOP_OPENMP
				//update the thread-local maximum wave speed
				l_maxWaveSpeed = std::max(l_maxWaveSpeed, maxEdgeSpeed);
			#else // LOOP_OPENMP
				//update the maximum wave speed
				maxWaveSpeed = std::max(maxWaveSpeed, maxEdgeSpeed);
			#endif // LOOP_OPENMP
	
		}
	*/
	#else
		for(int j = 1; j < ny_end; j++) {
			float maxEdgeSpeed;
			float hNetUpDow, hNetUpUpw;
			float hvNetUpDow, hvNetUpUpw;

			wavePropagationSolver.computeNetUpdates( h[i][j-1], h[i][j],
											   hv[i][j-1], hv[i][j],
											   b[i][j-1], b[i][j],
											   hNetUpDow, hNetUpUpw,
											   hvNetUpDow, hvNetUpUpw,
											   maxEdgeSpeed );

			// accumulate net updates to cell-wise net updates for h and hu
			hNetUpdates[i][j-1]  += dy_inv * hNetUpDow;
			hvNetUpdates[i][j-1] += dy_inv * hvNetUpDow;
			hNetUpdates[i][j]    += dy_inv * hNetUpUpw;
			hvNetUpdates[i][j]   += dy_inv * hvNetUpUpw;

			#ifdef LOOP_OPENMP
				//update the thread-local maximum wave speed
				l_maxWaveSpeed = std::max(l_maxWaveSpeed, maxEdgeSpeed);
			#else // LOOP_OPENMP
				//update the maximum wave speed
				maxWaveSpeed = std::max(maxWaveSpeed, maxEdgeSpeed);
			#endif // LOOP_OPENMP
		}
	#endif // VECTORIZE

	}

#ifdef LOOP_OPENMP
#ifdef USE_SCALASCA
  EPIK_USER_REG(r_name_crit, "WaveAccBlock_compNumFlux_crit");
  EPIK_USER_START(r_name_crit);  
#endif
	#pragma omp critical
	{
		maxWaveSpeed = std::max(l_maxWaveSpeed, maxWaveSpeed);
	}
#ifdef USE_SCALASCA
  EPIK_USER_END(r_name_crit);  
#endif
} // #pragma omp parallel
#endif
	
	if(maxWaveSpeed > 0.00001) {
		//TODO zeroTol

		//compute the time step width
		//CFL-Codition
		//(max. wave speed) * dt / dx < .5
		// => dt = .5 * dx/(max wave speed)

		maxTimestep = std::min( dx/maxWaveSpeed, dy/maxWaveSpeed );

		// reduce maximum time step size by "safety factor"
		maxTimestep *= (float) .4; //CFL-number = .5
	} else
		//might happen in dry cells
		maxTimestep = std::numeric_limits<float>::max();
		
#ifdef USE_SCALASCA
	EPIK_FUNC_END();
#endif
}

/**
 * Updates the unknowns with the already computed net-updates.
 *
 * @param dt time step width used in the update.
 */
void SWE_WaveAccumulationBlock::updateUnknowns(float dt) {

#ifdef USE_SCALASCA
	EPIK_FUNC_START();
#endif

  //update cell averages with the net-updates
#ifdef LOOP_OPENMP
	#pragma omp parallel for
#endif // LOOP_OPENMP
	for(int i = 1; i < nx+1; i++) {
#ifdef VECTORIZE
	h[i][1:ny] -= dt * hNetUpdates[i][1:ny];
	hu[i][1:ny] -= dt * huNetUpdates[i][1:ny];
	hv[i][1:ny] -= dt * hvNetUpdates[i][1:ny];

	hNetUpdates[i][1:ny] = (float) 0;
	huNetUpdates[i][1:ny] = (float) 0;
	hvNetUpdates[i][1:ny] = (float) 0;

	if (h[i][1:ny] < 0.1)
		hu[i][1:ny] = hv[i][1:ny] = 0.; //no water, no speed!


	if (h[i][1:ny] < 0) {
		#ifndef NDEBUG
			// Only print this warning when debug is enabled
			// Otherwise we cannot vectorize this loop
			if (h[i][1:ny] < -0.1) {
				std::cerr << "Warning, negative height: (i,j)=(" << i << ")=" << h[i][1:ny] << std::endl;
				std::cerr << "         b: " << b[i][1:ny] << std::endl;
			}
		#endif // NDEBUG
		//zero (small) negative depths
		h[i][1:ny] = (float) 0;
	}
#else
	for(int j = 1; j < ny+1; j++) {

		h[i][j]  -= dt * hNetUpdates[i][j];
		hu[i][j] -= dt * huNetUpdates[i][j];
		hv[i][j] -= dt * hvNetUpdates[i][j];

		hNetUpdates[i][j] = (float) 0;
		huNetUpdates[i][j] = (float) 0;
		hvNetUpdates[i][j] = (float) 0;

		//TODO: proper dryTol
		if (h[i][j] < 0.1)
			hu[i][j] = hv[i][j] = 0.; //no water, no speed!

		if (h[i][j] < 0) {
			#ifndef NDEBUG
				// Only print this warning when debug is enabled
				// Otherwise we cannot vectorize this loop
				if (h[i][j] < -0.1) {
					std::cerr << "Warning, negative height: (i,j)=(" << i << "," << j << ")=" << h[i][j] << std::endl;
					std::cerr << "         b: " << b[i][j] << std::endl;
				}
			#endif // NDEBUG
			//zero (small) negative depths
			h[i][j] = (float) 0;
		}
	}
#endif // VECTORIZE

	}
	
#ifdef USE_SCALASCA
	EPIK_FUNC_END();
#endif
}

/**
 * Update the bathymetry values with the displacement corresponding to the current time step.
 *
 * @param i_asagiScenario the corresponding ASAGI-scenario
 */
#ifdef DYNAMIC_DISPLACEMENTS
bool SWE_WaveAccumulationBlock::updateBathymetryWithDynamicDisplacement(scenarios::Asagi &i_asagiScenario, const float i_time) {

#ifdef USE_SCALASCA
	EPIK_FUNC_START();
#endif
  if (!i_asagiScenario.dynamicDisplacementAvailable(i_time))
    return false;

  // update the bathymetry
  for(int i=0; i<=nx+1; i++) {
    for(int j=0; j<=ny+1; j++) {
      b[i][j] = i_asagiScenario.getBathymetryAndDynamicDisplacement( offsetX + (i-0.5f)*dx,
                                                                     offsetY + (j-0.5f)*dy,
                                                                     i_time
                                                                   );
    }
  }

  setBoundaryBathymetry();

#ifdef USE_SCALASCA
	EPIK_FUNC_END();
#endif
  return true;
}
#endif
